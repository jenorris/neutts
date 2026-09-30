"""Persistent NeuTTS server, OpenAI-compatible (/v1/audio/speech).

Mirrors the whisper.cpp/openai_bridge STT pattern: hermes already knows how
to talk to an OpenAI-compatible TTS endpoint (``tts.provider: openai`` +
``tts.openai.base_url``), so this server just needs to *speak that dialect*
— no hermes source changes.

Why a server at all: hermes's own ``tools/neutts_synth.py`` deliberately
loads NeuTTS fresh in a subprocess per call (~17s model load + ~6-11s
reference-voice encoding, paid on *every* utterance — see the
2026-09-18 changelog entry). This process loads the model and encodes the
reference voice ONCE at startup and keeps both resident, so a request only
pays for the actual generation — measured ~2.8s to first audio, ~4s total
for a short reply, down from ~33s.

Two response modes, matching what hermes's OpenAI TTS client actually sends:
  - ``response_format: "pcm"`` → true incremental streaming via NeuTTS's own
    ``infer_stream()`` (this is what ``tools/tts_streaming.py``'s
    ``OpenAIStreamer`` requests — hermes's *native* low-latency path).
  - anything else (mp3/wav/flac/opus — the one-shot ``text_to_speech_tool``
    path) → full ``infer()``, transcoded via ffmpeg if the request didn't
    ask for wav.

A single worker and lock serialize both voices for the whole generation
lifetime. Streaming buffers four chunks and disconnects cancel generation
before releasing model ownership. The codec retains decoding weights only.
"""

import asyncio
import io
import json
import os
import re
import unicodedata
import wave
import struct
import subprocess
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import torch
from jobs import stream_owned, run_owned

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

# Lyra's persona skills were written against MiniMax's TTS markup (bracketed
# non-verbal cues MiniMax renders as actual sound: "(breath)", "(sighs)" —
# and its "<#0.500#>" pause-duration syntax). NeuTTS has no equivalent and
# no way to interpret either — left in, they'd be read aloud as literal
# words/punctuation. Since these are TTS control syntax, not spoken content,
# the correct fallback for an engine that can't render them is to drop them
# silently (closest to MiniMax's own behavior minus the actual sound), not
# speak them as text. See 2026-09-18 changelog for the full writeup.
_MINIMAX_PAUSE_RE = re.compile(r"<#\s*[\d.]+\s*#>")
# Exact set from agent.personalities.minimax-hd-tts in lyra's config.yaml,
# plus a few close variants, so nothing in that vocabulary can leak through
# as literal text regardless of which personality is actually active.
_MINIMAX_INTERJECTION_RE = re.compile(
    r"\(\s*(?:laughs?|chuckle[s]?|giggle[s]?|coughs?|clear[- ]throat|groans?|"
    r"breath(?:ing)?|pant(?:ing|s)?|inhale[s]?|exhale[s]?|gasp(?:s|ing)?|"
    r"sniff[s]?|sigh(?:s|ing)?|snort[s]?|burp[s]?|lip[- ]smack(?:ing)?|"
    r"humming|hums?|hissing|hiss(?:es)?|emm|sneeze[s]?|moan(?:s|ing)?|"
    r"whisper(?:s|ing)?|pause)\s*\)",
    flags=re.IGNORECASE,
)


def _strip_pictographs(text: str) -> str:
    # espeak reads emoji by name ("grinning squinting face"); YUI prefixes emoji voice tags.
    return "".join(
        " " if unicodedata.category(c) == "So" or c in "\u200d\ufe0e\ufe0f" or 0x1F3FB <= ord(c) <= 0x1F3FF
        else c
        for c in text
    )


# Pre-rendered phrases (e.g. Lyra's filler lines rendered once with MiniMax, the voice the clone
# is made from): NEUTTS_PRERENDER_DIR/index.json maps normalized text -> wav file in that dir.
PRERENDER_DIR = os.environ.get("NEUTTS_PRERENDER_DIR", "")


def _prerender_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _load_prerendered(directory=PRERENDER_DIR) -> dict:
    if not directory:
        return {}
    index_path = os.path.join(directory, "index.json")
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"prerender index unreadable ({index_path}): {exc}", flush=True)
        return {}
    return {k: os.path.join(directory, v) for k, v in index.items()}


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_LINE_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*\u2022]\s+|\d+[.)]\s+|>\s*)", re.MULTILINE)
_MD_MARKER_RE = re.compile(r"[*`]+|(?<!\w)_+|_+(?!\w)")


def _strip_markdown(text: str) -> str:
    # Chat replies arrive with markdown; espeak would read the asterisks and backticks aloud.
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_LINE_PREFIX_RE.sub("", text)
    text = _MD_MARKER_RE.sub("", text)
    # A line break without its own punctuation (list items, headings) still ends a spoken phrase.
    return re.sub(r"(?<![.!?:;,\u2026])[ \t]*\n+", ". ", text)


def _strip_minimax_markup(text: str) -> str:
    text = _strip_markdown(_strip_pictographs(text))
    text = _MINIMAX_PAUSE_RE.sub(" ", text)
    text = _MINIMAX_INTERJECTION_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

REF_AUDIO = os.environ.get(
    "NEUTTS_REF_AUDIO", "voices/primary/ref.wav"
)
REF_TEXT_PATH = os.environ.get(
    "NEUTTS_REF_TEXT", "voices/primary/ref.txt"
)
BACKBONE_REPO = os.environ.get(
    "NEUTTS_BACKBONE", "models/neutts-air-BF16.gguf"
)
CODEC_REPO = os.environ.get("NEUTTS_CODEC", "neuphonic/neucodec")
DEVICE = os.environ.get("NEUTTS_DEVICE", "cuda")  # "cuda" (ROCm) or "cpu"
# NeuTTS only auto-infers eSpeak language from a recognized "neuphonic/..."
# repo id — a local .gguf path (needed to keep llama.cpp/HIP loading; see
# 2026-09-18 changelog) isn't recognized, so this must be set explicitly.
LANGUAGE = os.environ.get("NEUTTS_LANGUAGE") or None
SAMPLE_RATE = 24_000

# NeuTTS's own generation bakes in inter-sentence pauses well past natural
# conversational pacing — measured directly (2026-09-18): up to ~0.78s of
# dead air after a single period, vs a natural ~0.2-0.4s. This doesn't
# touch words, only caps silence between them.
MAX_SILENCE_S = float(os.environ.get("NEUTTS_MAX_SILENCE_S", "0.35"))
_SILENCE_RMS_THRESHOLD = 0.02
_SILENCE_WINDOW_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms analysis window

_state: dict = {}
_lock = asyncio.Lock()


def _cap_silence(samples: np.ndarray, max_gap_s: float = MAX_SILENCE_S) -> np.ndarray:
    """Compress any contiguous near-silent run down to at most max_gap_s."""
    win = _SILENCE_WINDOW_SAMPLES
    max_gap_samples = int(max_gap_s * SAMPLE_RATE)
    n = len(samples)
    if n == 0:
        return samples

    def _rms(chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0

    out = []
    i = 0
    while i < n:
        chunk = samples[i:i + win]
        if _rms(chunk) < _SILENCE_RMS_THRESHOLD:
            j = i
            while j < n and _rms(samples[j:j + win]) < _SILENCE_RMS_THRESHOLD:
                j += win
            run_len = j - i
            keep = min(run_len, max_gap_samples)
            out.append(samples[i:i + keep])
            i = j
        else:
            out.append(chunk)
            i += win
    return np.concatenate(out) if out else samples


def _apply_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """Pitch-preserving time-stretch — librosa's phase vocoder, not a naive
    resample (which would also shift pitch, unlike a real speed control)."""
    if abs(speed - 1.0) < 1e-3:
        return samples
    speed = max(0.5, min(2.0, speed))
    return librosa.effects.time_stretch(np.asarray(samples, dtype=np.float32), rate=speed)


def _postprocess(samples: np.ndarray, speed: float) -> np.ndarray:
    return _apply_speed(_cap_silence(samples), speed)


def _initialize():
    from runtime import SharedNeuTTS

    definitions = {"primary": {"audio": REF_AUDIO, "text_path": REF_TEXT_PATH,
                                "prerender_dir": PRERENDER_DIR}}
    extra = os.environ.get("NEUTTS_EXTRA_VOICE")
    if extra:
        definitions[extra] = {
            "audio": os.environ["NEUTTS_EXTRA_REF_AUDIO"],
            "text_path": os.environ["NEUTTS_EXTRA_REF_TEXT"],
            "prerender_dir": os.environ.get("NEUTTS_EXTRA_PRERENDER_DIR", ""),
        }
    tts = SharedNeuTTS(
        backbone_repo=BACKBONE_REPO, backbone_device="gpu" if DEVICE == "cuda" else "cpu",
        codec_repo=CODEC_REPO, codec_device=DEVICE, language=LANGUAGE, seed=SEED,
    )
    references = tts.references({k: v["audio"] for k, v in definitions.items()},
                                os.environ.get("NEUTTS_REFERENCE_CACHE", str(Path.home() / ".cache/neutts/references")))
    voices = {}
    for name, definition in definitions.items():
        voices[name] = {
            "ref_codes": references[name],
            "ref_text": Path(definition["text_path"]).read_text().strip(),
            "prerendered": _load_prerendered(definition["prerender_dir"]),
        }
    _state.update(tts=tts, voices=voices)
    # Encode, prefill, initial codec windows and repeated streaming windows are all
    # exercised before the HTTP listener is declared ready. Audio is discarded.
    with torch.inference_mode():
        for name, voice in voices.items():
            tts.cancel_event.clear()
            for _ in tts.infer_stream(
                "Your voice service is ready. I can help you now.",
                voice["ref_codes"], voice["ref_text"],
                temperature=TEMPERATURE, top_k=TOP_K,
            ):
                pass
            print(f"Voice warm-up complete: {name}", flush=True)
    _state["ready"] = True
    print("Shared NeuTTS server ready.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neutts-gpu")
    _state["executor"] = executor
    try:
        await asyncio.get_running_loop().run_in_executor(executor, _initialize)
        yield
    finally:
        _state["ready"] = False
        tts = _state.get("tts")
        if tts:
            tts.cancel_event.set()
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
        if tts:
            tts.backbone.close()
        _state.clear()


app = FastAPI(lifespan=lifespan)


def _float_to_pcm16(wav: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _pcm16_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    n_channels, bits = 1, 16
    byte_rate = rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


_FORMAT_MIME = {
    "wav": ("wav", "audio/wav"),
    "mp3": ("mp3", "audio/mpeg"),
    "flac": ("flac", "audio/flac"),
    "opus": ("ogg", "audio/ogg"),
    "aac": ("adts", "audio/aac"),
}


def _transcode(wav_bytes: bytes, response_format: str) -> tuple[bytes, str]:
    if response_format not in _FORMAT_MIME or response_format == "wav":
        return wav_bytes, "audio/wav"
    ffmpeg_fmt, mime = _FORMAT_MIME[response_format]
    # ffmpeg's default bitrate for mono 24kHz input lands at 32kbps with no
    # explicit -b:a — audibly compressed/muffled for speech. 128k is well
    # past what 24kHz mono needs; cheap insurance against ever hitting a
    # bitrate floor again.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "-",
         "-b:a", "128k", "-f", ffmpeg_fmt, "-"],
        input=wav_bytes, capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode to {response_format} failed: "
                            f"{proc.stderr.decode(errors='replace')[:300]}")
    return proc.stdout, mime


# Sampling: NeuTTS defaults (temperature 1.0, top_k 50, fresh seed per call) make each call a new
# prosody draw, so consecutive sentences can swing in tone. Lower randomness + a fixed seed steady it.
TEMPERATURE = float(os.environ.get("NEUTTS_TEMPERATURE", "0.7"))
TOP_K = int(os.environ.get("NEUTTS_TOP_K", "30"))
_seed_env = os.environ.get("NEUTTS_SEED", "1234").strip()
SEED = int(_seed_env) if _seed_env else None
# Long inputs drift within one generation; synthesize them in sentence/clause-sized pieces.
MAX_PIECE_CHARS = int(os.environ.get("NEUTTS_MAX_PIECE_CHARS", "140"))
PIECE_GAP_S = 0.12


def _split_for_tts(text: str) -> list[str]:
    if len(text) <= MAX_PIECE_CHARS:
        return [text]
    units: list[str] = []
    for sentence in re.split(r"(?<=[.!?\u2026])\s+", text):
        if len(sentence) <= MAX_PIECE_CHARS:
            units.append(sentence)
            continue
        for clause in re.split(r"(?<=[,;:\u2014\u2013])\s+", sentence):
            while len(clause) > MAX_PIECE_CHARS:
                cut = clause.rfind(" ", 0, MAX_PIECE_CHARS)
                cut = cut if cut > 0 else MAX_PIECE_CHARS
                units.append(clause[:cut])
                clause = clause[cut:].lstrip()
            units.append(clause)
    pieces: list[str] = []
    for unit in (u.strip() for u in units):
        if not unit:
            continue
        if pieces and len(pieces[-1]) + 1 + len(unit) <= MAX_PIECE_CHARS:
            pieces[-1] = f"{pieces[-1]} {unit}"
        else:
            pieces.append(unit)
    return pieces


def _gap() -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * PIECE_GAP_S), dtype=np.float32)


def _voice(voice_id):
    voice = _state.get("voices", {}).get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Unknown voice endpoint")
    return voice


def _sync_infer(text, voice, cancelled):
    tts = _state["tts"]
    tts.cancel_event = cancelled
    parts = []
    with torch.inference_mode():
        for i, piece in enumerate(_split_for_tts(text)):
            if cancelled.is_set():
                break
            if i:
                parts.append(_gap())
            parts.append(np.asarray(tts.infer(
                piece, voice["ref_codes"], voice["ref_text"],
                temperature=TEMPERATURE, top_k=TOP_K), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _format_generated_audio(wav, speed, response_format):
    wav = _postprocess(wav, speed)
    wav_bytes = _pcm16_to_wav(_float_to_pcm16(wav))
    return _transcode(wav_bytes, response_format)


async def _speech(body: dict, voice_id="primary"):
    voice = _voice(voice_id)
    text = _strip_minimax_markup(str(body.get("input") or ""))
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'input' text")
    response_format = str(body.get("response_format") or "mp3").lower()
    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0

    clip = voice["prerendered"].get(_prerender_key(text))
    if clip and os.path.exists(clip):
        with open(clip, "rb") as f:
            wav_bytes = f.read()
        if response_format == "pcm":
            with wave.open(io.BytesIO(wav_bytes)) as w:
                return Response(content=w.readframes(w.getnframes()), media_type="audio/pcm")
        try:
            audio_bytes, mime = await asyncio.to_thread(_transcode, wav_bytes, response_format)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=audio_bytes, media_type=mime)

    if response_format == "pcm":
        def produce(cancelled, emit):
            tts = _state["tts"]
            tts.cancel_event = cancelled
            with torch.inference_mode():
                for i, piece in enumerate(_split_for_tts(text)):
                    if cancelled.is_set():
                        return
                    if i and not emit(_float_to_pcm16(_gap())):
                        return
                    with closing(tts.infer_stream(
                        piece, voice["ref_codes"], voice["ref_text"],
                        temperature=TEMPERATURE, top_k=TOP_K,
                    )) as chunks:
                        for chunk in chunks:
                            if cancelled.is_set() or not emit(_float_to_pcm16(_postprocess(chunk, speed))):
                                return
        return StreamingResponse(
            stream_owned(_lock, _state["executor"], produce), media_type="audio/pcm")

    wav = await run_owned(_lock, _state["executor"],
                          lambda cancelled: _sync_infer(text, voice, cancelled))
    try:
        # CPU stretching and ffmpeg must not stall health checks, streaming
        # consumers, or the bounded GPU producer's queue on the HTTP event loop.
        audio_bytes, mime = await asyncio.to_thread(
            _format_generated_audio, wav, speed, response_format)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=audio_bytes, media_type=mime)


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def speech(request: Request):
    body = await request.json()
    return await _speech(body)


# One resident cloned voice; clients that list speakers (e.g. YUI) get it as "clone".
@app.get("/v1/audio/voices")
@app.get("/audio/voices")
async def voices():
    return {"object": "list", "data": [{"id": "clone", "object": "voice"}]}


@app.get("/health")
async def health():
    return {"ok": bool(_state.get("ready")), "device": DEVICE,
            "voices": list(_state.get("voices", {})), "busy": _lock.locked()}


@app.post("/internal/voices/{voice_id}/v1/audio/speech")
async def voice_speech(voice_id: str, request: Request):
    return await _speech(await request.json(), voice_id)


@app.get("/internal/voices/{voice_id}/health")
async def voice_health(voice_id: str):
    _voice(voice_id)
    return await health()
