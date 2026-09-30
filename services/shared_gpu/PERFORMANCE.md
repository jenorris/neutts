# Follow-up performance audit — 2026-09-29

Measured with RX 7900 XTX, Ryzen 7 3700X, 64 GiB host RAM, NeuTTS-Air BF16,
NeuCodec FP32, Torch 2.10.0+rocm7.0 and llama-cpp-python 0.3.35.
Other host services remained online. Small timing differences are directional;
this was not an isolated, thermally controlled benchmark.

## Shipped changes

1. **Skip unused GGUF array metadata for phoneme models.** The original reader
   reparsed 217,652 vocabulary entries, 217,652 token types and 151,387 merges.
   This took 9.37 seconds independently. Only emotion metadata is used afterward,
   and phoneme models reject emotions. BPE models still use the original reader.
   Cached restart-to-warmed-readiness fell from 25.75 seconds to 15.45 seconds;
   a second restart took 16.00 seconds. Imports still take about 5.8 seconds.
2. **Move response formatting off the HTTP event loop.** Generated nonstreaming
   silence processing, speed adjustment, WAV construction and ffmpeg transcoding
   now run in a background thread. Prerendered transcoding also runs off-loop.
   ffmpeg cost 160 ms in one measurement, and initial speed adjustment at 1.2x
   cost 132 ms (20 ms warm). GPU ownership is released before formatting.
   This improves scheduling/responsiveness rather than individual synthesis time.

Eleven unit tests pass, including phoneme/BPE metadata compatibility and a
deliberately blocked transcoder with concurrent HTTP health handling. Four PCM
comparisons across two voices remain byte-identical. WAV at 1.2x speed and MP3
at normal speed are byte-identical before/after on both endpoints.
Both warmed endpoints are healthy. Live health timing remained noisy: maximum
95.5 ms before versus 78.6 ms after, with p99 9.3 versus 10.8 ms. These samples
do not establish a general latency improvement; the unit test and thread
placement verify removal of direct event-loop blocking.

## Where request time goes

A representative 4.71-second streamed utterance took 1.314 seconds to synthesize,
with first audio at 341 ms:

| Stage | Time | Share |
|---|---:|---:|
| llama.cpp prefill/token generation | 1.105 s | 84% |
| Codec decoding, including transfers | 117 ms | 9% |
| CPU watermarking | 86 ms | 6.5% |
| Silence processing and PCM formatting | 2.5 ms | 0.2% |

The codec's steady window is 81 tokens. Its attention already dispatches to the
memory-efficient FP32 kernel; attention accounted for about 3.4% of measured
codec GPU time. Matrix multiplication accounted for about 56%, with convolutions
about 17%. Decoder-only work therefore has limited impact on total request time.

## Experiments left disabled

| Candidate | Observation | Decision |
|---|---|---|
| Decoder HIP graph replay | 8.78 to 8.67 ms/window; exact output; about 250 MB extra reserved memory | Gain negligible for modal GPU sharing |
| Torch CPU threads, GPU decoder | 1–16 threads gave similar 8.5–8.8 ms timings | Retain default |
| Watermarker CPU threads | 1 and 2 slower; 4 marginal; 16 severely slower; 8 remained best in final pass | Retain default; avoid oversubscription |
| ROCm TunableOp | 8.82 to 8.16 ms/window; retested eager 8.54 ms; max waveform difference 2.24e-6 | Roughly sub-1% whole-request opportunity, numerically different |
| `torch.compile` decoder | 9.05 to 7.50 ms/window; first compile 18.4 s; max waveform difference 9.98e-7 | About 1–2% whole-request estimate; more work needed for varying shapes/cold starts |
| Forced math attention | Slower and numerically different | Keep efficient FP32 attention |
| Forced FlashAttention | No available kernel for FP32 input; requires Half/BFloat16 | Preserve decoder precision |
| llama.cpp threads 1/2/4/8/16 | Typically around 0.68–0.70 s for 128 generated tokens; output identical | No robust gain established |
| llama.cpp batches 128/256/1024 | Slower or similar versus 512; changed sampled output | Keep 512 |
| Explicit operation offload | No gain; output identical | Keep existing default |
| llama.cpp flash attention disabled | Slightly slower and changed output | Keep enabled |
| `GGML_CUDA_GRAPH_OPT=1` | Slower/unstable timings and changed output | Keep disabled |

No model quantization, mixed precision, TF32, model substitution, watermark
removal or sampling change was introduced. `torch.compile` and tuning experiments
were confined to separate processes and did not modify production settings.

## Remaining opportunities, in priority order

1. **Isolated newer Torch/ROCm environment test.** For Python 3.11, official
   wheel indexes currently provide Torch 2.13.0+rocm7.1 and 2.14.0+rocm7.2.
   ROCm 7.0's latest wheel remains 2.10.0. Check host driver compatibility and
   matched torchaudio/torchtune/torchao dependencies before evaluating. Avoid
   upgrading the shared agent environment as an experiment. Torch changes affect
   the codec and watermark stack; llama.cpp has its own compiled HIP backend.
2. **Bounded audio response cache for repeated phrases.** A fixed seed allows
   identical requests to bypass generation entirely. Bind entries to model,
   reference voice/transcript, all synthesis/postprocessing settings and format;
   respect unseeded requests and impose a memory limit. Existing prerenders already
   cover some phrases. Benefits depend on actual repeat frequency, not benchmark
   token throughput. Not implemented in this audit.
3. **Overlap CPU watermarking with generation of the next chunk.** CPU
   watermarking currently serializes about 86 ms across ten chunks. A bounded
   ordered pipeline could recover some of that time; cancellation, backpressure,
   deterministic output and worker teardown need dedicated validation. The 6.5%
   stage share is an upper bound, not a measured improvement. Not implemented.
4. **Same-input llama.cpp prefix reuse.** Upstream resets explicitly to preserve
   seeded reproducibility. Do not remove the reset casually. Changing target text
   limits reusable prefixes, and state restoration/caching would need exact audio
   tests and a RAM budget. Full response caching is simpler for identical inputs.
5. **Compiler/tuning follow-up only if throughput matters.** Test all streaming
   and final shapes, nonstream lengths, startup impact, VRAM and numerical/audio
   quality. Present gains are too small to enable by default.

Checkpoint construction takes about 1.1–1.6 seconds. Meta/assign loading could
reduce initialization, but NeuCodec creates nonpersistent quantizer/RoPE buffers
and performs `.item()` during construction, making a blanket meta-device switch
unsafe. Low priority after eliminating the 9.4-second redundant GGUF read.

## Primary references

- [Official ROCm 7.0 Torch wheels](https://download.pytorch.org/whl/rocm7.0/torch/)
- [Official ROCm 7.1 Torch wheels](https://download.pytorch.org/whl/rocm7.1/torch/)
- [Official ROCm 7.2 Torch wheels](https://download.pytorch.org/whl/rocm7.2/torch/)
- [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [AMD TunableOp tuning guidance](https://rocm.docs.amd.com/en/docs-7.0.2/how-to/rocm-for-ai/inference-optimization/workload.html)
- [PyTorch checkpoint loading guidance](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)
- [llama.cpp graph implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)
