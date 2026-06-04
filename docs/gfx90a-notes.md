## gfx90a / MI210 notes

- MI210 is `gfx90a` with 104 CUs; set `GPU_ARCHS=gfx90a` (and `CU_NUM=104` if needed) so AOT/FlyDSL does not build MI300/MI350 kernels.
- `gfx90a` can use FP8 storage/conversion paths, but it has no FP8 matrix-compute ISA. Gate FP8 GEMM kernels on `is_fp8_compute_available()`.
- AITER builds for `gfx90a` must exclude newer ISA families (FP4/MXFP4, DeepGEMM/MLA, FMHA v3, asm/blockscale paths) unless explicitly ported.
- Qwen reasoning models need chat templating. Use `/v1/chat/completions`; for clean content by default, start vLLM with `--default-chat-template-kwargs '{"enable_thinking":false}'`.
- On vLLM ROCm, AITER can be enabled globally while selectively disabling risky subpaths, e.g. `VLLM_ROCM_USE_AITER=1` with MoE/MHA/RMSNorm/Linear tested one-by-one.
