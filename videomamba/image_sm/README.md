# Image Classification

## VideoViT Architecture Comparison

`models/videovit.py` provides independent `videovit_tiny`, `videovit_small`,
`videovit_middle`, and `videovit_base` models. They preserve the corresponding
VideoMamba patch embedding, width, depth, RMSNorm, residual path, position
embedding, and classification head. The bidirectional Mamba sequence mixer is
replaced by multi-head scaled dot-product softmax attention, followed by the
reference bias-free slow SwiGLU MLP. Its hidden ratio is configurable with
`--mlp-ratio` and defaults to 3.

Create the tested environment from the repository's `videomamba` directory, or
activate the already-created local environment:

```shell
conda env create -f environment-videovit.yml
conda activate videovit
```

### ImageNet-1K data

The training loader expects filesystem images plus metadata lines in
`relative/image/path.JPEG integer_label` format. The Hugging Face
`ILSVRC/imagenet-1k` dataset is gated, so accept its terms and authenticate with
`hf auth login` before downloading the labeled splits:

```shell
export HF_HOME=/mnt/localssd/dataset/videovit/.hf_home
hf download ILSVRC/imagenet-1k \
    --repo-type dataset \
    --include 'data/train-*.parquet' \
    --include 'data/validation-*.parquet' \
    --include classes.py \
    --include README.md \
    --local-dir /mnt/localssd/dataset/videovit/imagenet-1k-hf \
    --max-workers 16

python prepare_imagenet_hf.py \
    --source /mnt/localssd/dataset/videovit/imagenet-1k-hf \
    --output /mnt/localssd/dataset/videovit/imagenet-1k \
    --workers 8
```

The converter is resumable at shard granularity. Train with:

```shell
--root_dir_train /mnt/localssd/dataset/videovit/imagenet-1k/train \
--meta_file_train /mnt/localssd/dataset/videovit/imagenet-1k/meta/train.txt \
--root_dir_val /mnt/localssd/dataset/videovit/imagenet-1k/val \
--meta_file_val /mnt/localssd/dataset/videovit/imagenet-1k/meta/val.txt
```

Use the existing training commands with a VideoViT model name, for example:

```shell
python main.py --model videovit_tiny <your existing ImageNet arguments>
```

VideoViT does not currently provide pretrained checkpoints; train from scratch
or pass a compatible checkpoint with `--finetune`.

To monitor image training in the public `LVSM-Experiment/videosft` W&B project,
explicitly clear the machine's internal endpoint and provide the run name for
the current experiment:

```shell
unset WANDB_BASE_URL
python main.py \
    --model videovit_tiny \
    --wandb \
    --wandb-run-name <run-name> \
    <your existing ImageNet arguments>
```

Only distributed rank 0 creates a W&B run. The code never prints or clears the
W&B API key.

We currenent release the code and models for:

- [x] **ImageNet-1K pretraining**

- [x] **Large resolution fine-tuning**



## Update

- :fire: **03/12/2024**: Pretrained models on ImageNet-1K are released.



## Model Zoo

See [MODEL_ZOO](./MODEL_ZOO.md).


## Usage

### Normal Training

Simply run the training scripts in [exp](exp) as followed:

```shell
bash ./exp/videomamba_tiny/run224.sh
```

> If the training was interrupted abnormally, you can simply rerun the script for auto-resuming. Sometimes the checkpoint may not be saved properly, you should set the resumed model via `--reusme ${OUTPUT_DIR}/ckpt/checkpoint.pth`.

### Training w/ SD

Simply run the training scripts in [exp_distill](exp_distill) as followed:

```shell
bash ./exp_distill/videomamba_middle/run224.sh
```

> For `teacher_model`, we use a smaller model by default.

### Large Resolution Fine-tuning

Simply run the training scripts in [exp](exp) as followed:

```shell
bash ./exp/videomamba_tiny/run448.sh
```

> Please set pretrained model via `--finetune`.

### Evaluation

Simply add `--eval` in the training scripts.

> It will evaluate the last model by default. You can set other models via `--resume`.

### Generate curves

You can generate the training curves as followed:

```shell
python3 generate_tensoboard.py
```

Note that you should install `tensorboardX`.
