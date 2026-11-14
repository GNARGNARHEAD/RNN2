# Retired local GPU feasibility experiment

Measured 4 November 2026; retired from the active project on 14 November 2026 when GNARGNARHEAD selected the approximately 20B student on one rented B200. This record preserves historical evidence, not a current build or training path.

## What was established

A disposable C++/LibTorch model trained from seeded random weights on an AMD Ryzen 5 6600H integrated GPU (`gfx1035`). The backend was ROCm 10.1.0 / PyTorch 2.14.0. Model parameters, gradients, and AdamW state were FP32; attention used explicit causal matrices. The workload used a 256-symbol vocabulary and generally batch 4, sequence length 128.

- Models with 27,872 and 3,320,576 parameters passed CPU/GPU comparisons of forward tensors, gradients, and optimizer updates, plus a causal-prefix check and a controlled parameter edit.
- A simple synthetic prediction loss decreased. Model/optimizer reload reproduced the next update on the same device. RNG, tutor history, and full learning-position recovery were not tested.
- A 151,775,488-parameter model completed three measured minutes at about 719 synthetic positions/second. Available host memory reached about 1.98 GiB; GPU temperature reached 93°C and CPU temperature 89.1°C. No overnight result was established.

These results showed limited numerical feasibility. They did not establish educational capability, retention, transfer, long-run stability, or a general model-size ceiling. They do not predict B200 performance or validate compressed-state FP8/FP4 training.

## What remains useful

Carry the verification questions into the B200 implementation: actual forward/backward/update correctness, causality, inspectability, controlled edits, measured peak resources, and continued updates after recovery. The new runtime must additionally preserve random/numerical state and the learner's educational position.

The old source, build scripts, complete experiment report, and reproduction commands remain in Git history at revision `0d91a9ce4de5752006005bb1ce61e122ccaed74a`. They were removed from the current checkout rather than maintained as a second runtime. Local raw measurements and disposable verification checkpoints remain under ignored `runs/gpu-probe-2026-11-04/` for provenance. The generated local SDK and build cache were removed during retirement.

The current work is defined by the [B200 implementation and checkpoint plan](../2026-11-14-b200-plan.md) and [project direction](../../README.md).
