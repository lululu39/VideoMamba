# Repository Notes

## VideoViT Comparison Models

- The image model lives in `videomamba/image_sm/models/videovit.py` and
  registers `videovit_tiny`, `videovit_small`, `videovit_middle`, and
  `videovit_base`.
- The video model lives in `videomamba/video_sm/models/videovit.py` and
  registers `videovit_tiny`, `videovit_small`, and `videovit_middle`.
- These are independent models. Do not add an attention mode or attention flag
  to the existing VideoMamba classes.
- The comparison intentionally preserves the corresponding VideoMamba patch or
  tubelet embedding, width, depth, RMSNorm, delayed residual path, stochastic
  depth, positional embeddings, and classification head.
- Each bidirectional Mamba sequence mixer is replaced with multi-head scaled
  dot-product softmax attention implemented through PyTorch SDPA. There is no
  MLP branch, so the controlled experiment changes only the sequence mixer.
- Tiny, small, middle, and base use 3, 6, 9, and 12 attention heads
  respectively, keeping a 64-dimensional head.

## Environment

- The tested Conda environment is named `videovit`.
- Its reproducible specification is `videomamba/environment-videovit.yml`.
- The validated core stack is Python 3.11, PyTorch 2.13.0 with CUDA 13.0,
  torchvision 0.28.0, timm 1.0.28, and NumPy 2.4.6.
- Mamba CUDA extensions, Apex, DeepSpeed, TensorFlow, and xFormers are not
  required for the VideoViT image classification or video classification and
  regression paths.

## Compatibility Changes

- Both model packages treat `mamba_ssm` as optional, allowing VideoViT training
  without importing the Mamba CUDA extension. Import failures unrelated to
  `mamba_ssm` must still be raised.
- Video classification and regression route `videovit_*` through the same
  image size, tubelet size, frame count, checkpointing, and separate
  spatial/temporal checkpoint interpolation interface used by VideoMamba.
- The obsolete and unused `numpy.lib.function_base.disp` imports were removed
  for NumPy 2.x compatibility.
- `optim_factory.py` uses the public timm optimizer exports and `NAdam` name,
  and maps `layers.N` parameters to their actual layer-decay depth.
- VideoViT factories remove timm-injected model metadata keys explicitly. Do
  not replace this with a model constructor that silently accepts arbitrary
  keyword arguments.

## Validation Performed

- Image and video registry construction through `timm.create_model`.
- CPU FP32 and H100 GPU BF16 forward and backward passes.
- Video gradient checkpointing.
- Full `224x224` image and `8x224x224` video inference with tiny models.
- Numerical equivalence between SDPA and explicit softmax attention.
- Spatial and temporal position-embedding interpolation followed by strict
  checkpoint loading.
- AdamW and NAdam creation, layer-decay assignment, compileall, Conda file dry
  run, and Git whitespace checks.

## Current Git State

- The VideoViT work is committed locally on `main` and is intentionally not
  pushed per the repository owner's request.
