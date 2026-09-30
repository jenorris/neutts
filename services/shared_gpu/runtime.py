"""Shared NeuTTS runtime, preserving the installed BF16 synthesis algorithms."""
from contextlib import closing
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import threading

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from neucodec import NeuCodec
from neucodec.codec_decoder_vocos import CodecDecoderVocos
from neutts import NeuTTS
from neutts.neutts import _normalize_text


def cached_file(repo, name):
    try:
        return hf_hub_download(repo, name, local_files_only=True)
    except FileNotFoundError:
        return hf_hub_download(repo, name)


class DecoderOnlyCodec(torch.nn.Module):
    """The exact decoder weights and operations used by NeuCodec.decode_code."""
    def __init__(self, checkpoint):
        super().__init__()
        self.generator = CodecDecoderVocos(hop_length=480)
        self.fc_post_a = torch.nn.Linear(2048, 1024)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        decoder_state = {k: v for k, v in state.items()
                         if k.startswith(('generator.', 'fc_post_a.'))}
        # The shipped checkpoint retains these removed legacy decoder modules;
        # upstream NeuCodec ignores them with strict=False. Keep every active
        # decoder tensor mandatory rather than hiding any other mismatch.
        legacy = ('generator.backbone.embed.', 'generator.backbone.prior_net.',
                  'generator.backbone.post_net.')
        expected = set(self.state_dict())
        extra = set(decoder_state) - expected
        if any(not (k.startswith(legacy) and k.endswith(('.weight_g', '.weight_v')))
               for k in extra):
            raise ValueError(f'Unexpected decoder checkpoint tensors: {sorted(extra)}')
        self.load_state_dict({k: v for k, v in decoder_state.items() if k in expected}, strict=True)

    @property
    def device(self):
        return self.fc_post_a.weight.device

    def decode_code(self, codes):
        emb = self.generator.quantizer.get_output_from_indices(codes.transpose(1, 2))
        emb = emb.transpose(1, 2)
        emb = self.fc_post_a(emb.transpose(1, 2)).transpose(1, 2)
        return self.generator(emb.transpose(1, 2), vq=False)[0]


class RollingOverlap:
    """Accumulate the same weighted overlap as upstream, retaining only the tail."""
    def __init__(self, stride):
        self.stride = stride
        self.out = None
        self.weight = None

    def add(self, frame, final=False):
        if self.out is None:
            self.out = np.zeros(frame.shape, dtype=frame.dtype)
            self.weight = np.zeros(frame.shape[-1], dtype=frame.dtype)
        elif frame.shape[-1] > self.out.shape[-1]:
            extra = frame.shape[-1] - self.out.shape[-1]
            self.out = np.pad(self.out, [(0, 0)] * (frame.ndim - 1) + [(0, extra)])
            self.weight = np.pad(self.weight, (0, extra))
        t = np.linspace(0, 1, frame.shape[-1] + 2, dtype=frame.dtype)[1:-1]
        weight = 0.5 - np.abs(t - 0.5)
        self.out[..., :frame.shape[-1]] += weight * frame
        self.weight[:frame.shape[-1]] += weight
        n = self.out.shape[-1] if final else self.stride
        if np.any(self.weight[:n] <= 0):
            raise ValueError('Uncovered audio overlap')
        result = self.out[..., :n] / self.weight[:n]
        self.out = self.out[..., n:].copy()
        self.weight = self.weight[n:].copy()
        return result


class SharedNeuTTS(NeuTTS):
    def __init__(self, *args, **kwargs):
        self.cancel_event = threading.Event()
        self._reference_tokens = {}
        super().__init__(*args, **kwargs)

    def _read_gguf_array_meta(self, model_path):
        # Only BPE models accept emotions. Parsing every tokenizer array again
        # costs seconds for Air's large vocabulary and provides no used metadata.
        if self.input_format == 'phonemes':
            return {}
        return super()._read_gguf_array_meta(model_path)

    def _load_codec(self, repo, device):
        if repo != 'neuphonic/neucodec':
            raise ValueError('Shared runtime currently supports the original NeuCodec only')
        self.codec_repo = repo
        self.codec_checkpoint = cached_file(repo, 'pytorch_model.bin')
        self.codec = DecoderOnlyCodec(self.codec_checkpoint).eval().to(device)

    def references(self, audio_paths, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Bind cached codes to the audio and both sets of encoder weights.
        semantic = cached_file('facebook/w2v-bert-2.0', 'model.safetensors')
        identities = []
        for path in [self.codec_checkpoint, semantic]:
            p = Path(path).resolve()
            stat = p.stat()
            identities.append((str(p), stat.st_size, stat.st_mtime_ns))
        signature = {'version': 1, 'weights': identities,
                     'neucodec': importlib.metadata.version('neucodec')}
        paths = {}
        values = {}
        missing = []
        for voice, audio_path in audio_paths.items():
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()
                                 + Path(audio_path).read_bytes()).hexdigest()
            paths[voice] = cache_dir / (key + '.json')
            try:
                codes = json.loads(paths[voice].read_text())
                if not isinstance(codes, list) or not codes or not all(type(v) is int and v >= 0 for v in codes):
                    raise ValueError('Invalid reference code cache')
                values[voice] = codes
                print(f'Reference code cache hit: {voice}', flush=True)
            except (OSError, ValueError):
                missing.append(voice)
        if missing:
            # Reference encoding is startup-only; never retain its modules on GPU.
            decoder = self.codec
            encoder = NeuCodec.from_pretrained(self.codec_repo).eval().to(decoder.device)
            try:
                self.codec = encoder
                for voice in missing:
                    codes = super().encode_reference(audio_paths[voice]).detach().cpu().tolist()
                    values[voice] = codes
                    temporary = paths[voice].with_suffix('.tmp')
                    temporary.write_text(json.dumps(codes))
                    os.replace(temporary, paths[voice])
                    print(f'Reference encoded and cached: {voice}', flush=True)
            finally:
                self.codec = decoder
                del encoder
                if decoder.device.type == 'cuda':
                    torch.cuda.empty_cache()
        for codes in values.values():
            self._reference_tokens[tuple(codes)] = tuple(f'<|speech_{i}|>' for i in codes)
        return values

    @lru_cache(maxsize=256)
    def _to_phones(self, text):
        return super()._to_phones(text)

    def _ggml_prompt(self, ref_codes, ref_text, input_text, emotion=None):
        codes = ''.join(self._reference_tokens[tuple(ref_codes)])
        if self.input_format == 'phonemes':
            text = self._to_phones(ref_text) + ' ' + self._to_phones(input_text)
            return (f'user: Convert the text to speech:<|TEXT_PROMPT_START|>{text}'
                    f'<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes}')
        text = _normalize_text(ref_text)
        target = _normalize_text(input_text)
        if emotion is None:
            text += ' ' + target
        else:
            emotion_token = f'<|{emotion.upper()}|>'
            if len(self.backbone.tokenize(emotion_token.encode(), add_bos=False, special=True)) != 1:
                raise ValueError(f'Emotion token {emotion_token} is not in the model vocab')
            text += emotion_token + target
        return f'<|TEXT_PROMPT_START|>{text}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes}'

    def _infer_stream_ggml(self, ref_codes, ref_text, input_text, emotion=None,
                          temperature=1.0, top_k=50):
        prompt = self._ggml_prompt(ref_codes, ref_text, input_text, emotion)
        self.backbone.reset()  # Preserve seeded full-prefill behavior.
        token_cache = list(self._reference_tokens[tuple(ref_codes)])
        n_decoded_tokens = len(ref_codes)
        overlap = RollingOverlap(self.streaming_stride_samples)
        with closing(self.backbone(prompt, max_tokens=self.max_context,
                                  temperature=temperature, top_k=top_k,
                                  stop=['<|SPEECH_GENERATION_END|>'], stream=True,
                                  seed=self._call_seed())) as tokens:
            for item in tokens:
                if self.cancel_event.is_set():
                    return
                token_cache.append(item['choices'][0]['text'])
                if len(token_cache) - n_decoded_tokens >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                    start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                    end = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                    recon = self._decode(''.join(token_cache[start:end]))
                    if self.watermarker is not None:
                        recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
                    sample_start = (n_decoded_tokens - start) * self.hop_length
                    sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                    yield overlap.add(recon[sample_start:sample_end])
                    n_decoded_tokens += self.streaming_frames_per_chunk
        remaining = len(token_cache) - n_decoded_tokens
        if remaining and not self.cancel_event.is_set():
            start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining), 0)
            sample_start = (len(token_cache) - start - remaining - self.streaming_overlap_frames) * self.hop_length
            recon = self._decode(''.join(token_cache[start:]))
            if self.watermarker is not None:
                recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            yield overlap.add(recon[sample_start:], final=True)
