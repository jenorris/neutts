# Shared GPU NeuTTS service

Maintained on `shared-gpu-runtime` in https://github.com/jenorris/neutts.
Upstream: https://github.com/neuphonic/neutts, baseline commit
`ac69851f28fc63a487917e7c2e27f0d75c759cba` (NeuTTS 1.4.1).
The baseline `neutts/neutts.py` exactly matches the installed source used in
the before/after validation. Upstream package APIs and licensing are retained.

## Changes

The runtime is an opt-in `SharedNeuTTS` subclass, with an OpenAI-compatible
FastAPI service and a lightweight proxy for a second voice endpoint:

- Decoder-only NeuCodec: load all active decoder tensors strictly; temporarily
  construct the original reference encoder only when a cache needs filling.
- Persistent CPU integer reference codes, cached prompt tokens and phonemes.
- Exclusive model ownership for the complete request/stream lifetime.
- Cancellation waits for producer shutdown; streaming queue bounds four chunks.
- Full synthesis warmup for each configured voice before readiness.
- One worker serves both voices, with no duplicate model in the proxy.
- Rolling overlap-add preserves upstream output while retaining only the tail.
- Phoneme models skip unused GGUF array metadata parsing; BPE emotion metadata remains intact.
- CPU response processing and transcoding run outside the HTTP event loop.

Backbone BF16 and decoder FP32 are unchanged. Seed, generation settings,
watermarking, splitting, and audio postprocessing are preserved. This runtime
currently supports the original `neuphonic/neucodec` codec only. The internal
voice API assumes a trusted loopback deployment; both voices share availability
and a serial queue.

## Setup

Use Python 3.11 with the appropriate GPU stack already installed. Tested with
PyTorch/torchaudio 2.10.0+rocm7.0 and llama-cpp-python 0.3.35 built with HIP and
flash attention on gfx1100. Preserve these GPU builds when installing:

```sh
python -m pip install --no-deps .
python -m pip install -r services/shared_gpu/requirements.txt
```

NeuTTS's remaining dependencies are declared in the root `pyproject.toml`.
This is an environment recipe, not a complete dependency lock. Also install
espeak-ng and ffmpeg. Model weights and reference recordings are not included.

Set `NEUTTS_BACKBONE` to a local BF16 GGUF, `NEUTTS_REF_AUDIO` to reference WAV,
and `NEUTTS_REF_TEXT` to its transcript file. Set `NEUTTS_DEVICE=cuda` for ROCm
and `NEUTTS_LANGUAGE=en-us` for the tested English setup. For another voice,
set `NEUTTS_EXTRA_VOICE=secondary`, `NEUTTS_EXTRA_REF_AUDIO`, and
`NEUTTS_EXTRA_REF_TEXT`.

```sh
cd services/shared_gpu
python -m uvicorn server:app --host 127.0.0.1 --port 8290
# In another terminal, using the same environment:
NEUTTS_PROXY_VOICE=secondary python -m uvicorn proxy:app --host 127.0.0.1 --port 8291
```

Both speech aliases `/v1/audio/speech` and `/audio/speech` are supported.
PCM streams; WAV and ffmpeg formats use complete generation. `/health` becomes
available after full startup warmup and reports worker ownership via `busy`.
Reference caches default to `~/.cache/neutts/references`, overridable through
`NEUTTS_REFERENCE_CACHE`. Identities bind the audio bytes to encoder checkpoint
paths, sizes, mtimes, and codec version. A cache miss rebuilds that reference.

`owner.conf` and `voice-proxy.conf` are example systemd **drop-ins**, assuming
existing `neutts-server.service` and `neutts-server-desma.service` units.
Adjust paths and provide the primary model/reference environment in the base
unit or another drop-in. Restarting the worker restarts a running proxy.
Stop and start both unit names for modal transitions. Rollback: stop both,
remove these drop-ins, daemon-reload, then start the original units.

## Validation recorded on 2026-09-29

RX 7900 XTX (24 GiB), 64 GiB host RAM:

- Combined NeuTTS VRAM: 11.52 GiB to 3.01 GiB, 2.97 GiB on cached restart.
- Warm first audio: approximately 0.25–0.40 seconds.
- Four before/after streamed PCM outputs were byte-identical across two voices.
- Concurrent clients, cancellation through direct and proxied endpoints, WAV,
  MP3, and cached startup passed. Cached restart-to-readiness took 25.75 seconds.

```sh
cd services/shared_gpu
python -m unittest -v test_runtime
```

Eleven tests cover exact rolling audio equivalence (including short final windows),
cache lifecycle, overlapping streams, disconnect/task cancellation, backpressure,
and failure cleanup. Ownership tests use two executor threads deliberately.

## Maintenance

Keep `upstream` pointed at Neuphonic and merge upstream changes into this branch
deliberately. Rerun tests and live audio comparisons after updates to upstream
prompt/streaming code, NeuCodec, or GPU libraries. Update checkpoint compatibility
checks if decoder architecture changes; do not hide missing active weights with
`strict=False`. The host's original workspace service directory remains the
active deployment; this fork preserves its code with generic defaults/templates.

The follow-up [performance audit](PERFORMANCE.md) records additional experiments
and the startup/responsiveness improvements shipped after the initial tag.
