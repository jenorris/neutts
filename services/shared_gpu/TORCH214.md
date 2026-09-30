# Torch 2.14 / ROCm 7.2 trial — 2026-09-30

Tested the highest available Python 3.11 ROCm 7.2 Torch wheel,
`torch==2.14.0+rocm7.2`, against the live stack's `2.10.0+rocm7.0` on an RX
7900 XTX. Model and reference inputs were unchanged: BF16 NeuTTS-Air backbone,
FP32 decoder, highest FP32 matmul precision, original watermarking and seed.
No quantization or mixed precision was introduced. The live environment was not
upgraded.

## Results

Two eager passes per stack, four voice/text cases per pass, three measured
repetitions per case after warmup; 40 measured decoder calls per shape/pass.
Other host services remained available, so small differences are not statistically
established gains. In particular, the initial apparent request improvement did
not persist when the baseline was repeated.

| Measure | Torch 2.10 / ROCm 7.0 | Torch 2.14 / ROCm 7.2 |
|---|---:|---:|
| Steady decoder window, median range | 8.70–8.74 ms | 8.48–8.62 ms |
| Final decoder window, median range | 8.72–9.67 ms | 8.70–8.96 ms |
| Sum of four pooled request medians | 5.046 s | 5.065 s |
| PyTorch GPU allocated memory | 826.9 MiB | 750.9 MiB |
| PyTorch GPU reserved memory | 840 MiB | 764 MiB |

Eager decoding is about 1–3% faster for the steady shape. Complete request time
is effectively unchanged (0.37% slower in the pooled summary). First-audio latency
is also not improved consistently. Torch-managed GPU memory falls by about
76 MiB; these counters exclude llama.cpp and allocations outside PyTorch.

All eleven service tests passed. Four complete PCM outputs retained their exact
lengths. Maximum difference versus the old stack is one int16 least-significant
bit, affecting approximately 0.3–1.0% of samples. Raw decoder maximum absolute
differences were 2.11e-6 and 8.79e-7 for the two tested shapes. Output is therefore
numerically slightly different, not byte-identical across Torch versions.

## New compiler

With `torch.compile(codec.decode_code, fullgraph=True, dynamic=True)` and the
same full precision settings:

- Steady window: 7.13 ms; final window: 7.00 ms.
- Initial compilation: 23.77 seconds, excluding subsequent shape handling.
- Sum of four request medians: 5.007 seconds, about 0.8% faster than the pooled
  old-stack summary. This small end-to-end difference remains vulnerable to host
  noise; improved decoder timing does not demonstrate a large service gain.
- First-audio latency did not improve consistently.
- Compilation remained an experiment; no compiled decoder was deployed.

Most request time remains in llama.cpp generation. Changing Torch does not
rebuild that backend. Runtime library loading can also interact between GPU
packages in one process, so these measurements evaluate the actual combined
application rather than assuming complete library independence.

## Dependency setup

The official ROCm 7.2 index supplied Torch 2.14, but its newest torchaudio wheel
was 2.11. A `torchaudio==2.14.0` wheel was unavailable there, on the CPU index,
or on PyPI. The test therefore built upstream torchaudio source against the
installed Torch 2.14 headers rather than forcing an old binary into the test.

- Torch wheel: official `2.14.0+rocm7.2`, CPython 3.11, x86_64.
- Wheel SHA256: `95ff2c8cc86b5e950ca55b2e8272c60d70c138851869bcca5eee6a36710ef9e0`.
- Bundled HIP runtime: 7.2.53211.
- Triton: `triton-rocm~=3.8.0` (installed 3.8.0).
- Torchaudio source: upstream commit beginning `245ccb4`, built version
  `2.11.0a0+245ccb4`, compiled against Torch 2.14 using the stable extension ABI.
- Build: `USE_ROCM=0 USE_CUDA=0 BUILD_RNNT=0 BUILD_ALIGN=0
  BUILD_CUDA_CTC_DECODER=0 MAX_JOBS=2`, CPU extension; ordinary ATen tensor
  operations dispatch through the installed Torch backend.

NeuCodec imports and complete cached-reference voice synthesis succeeded. The
existing torchao installation emitted warnings about unused CUDA extensions;
they did not prevent this FP32 runtime or its tests. Fresh GPU reference encoding
with an empty real voice cache was not included in this trial. A permanent
upgrade should use a dedicated environment and resolve the optional-extension
warnings, reference-cache miss path, and package support before deployment.

The test stack lived in temporary storage because the root filesystem lacked
space for another full installation. No production service pointed at it.
The source-build recipe and small torchaudio wheel were saved with local test
artifacts; large temporary binaries/downloads can be removed after measurement.

## Decision

Keep the current live stack. Torch 2.14/ROCm 7.2 runs and reduces decoder memory,
but the repeated measurements do not justify the expected large inference gain
or the additional deployment/dependency work for this service alone. It remains
a valid candidate for a broader GPU software refresh or another workload with
a larger PyTorch share.

Official wheel inventories: [Torch ROCm 7.2](https://download.pytorch.org/whl/rocm7.2/torch/),
[torchaudio ROCm 7.2](https://download.pytorch.org/whl/rocm7.2/torchaudio/).
Source build: [PyTorch audio](https://github.com/pytorch/audio).
