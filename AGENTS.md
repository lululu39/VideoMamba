# 仓库开发说明

## 总体目标

- 本仓库用于比较 VideoMamba、VideoViT 和 VideoLACT 在图像预训练、视频监督训练下的架构能力。
- 新架构必须作为独立模型实现。不要在 VideoMamba 或 VideoViT 中加入切换 mixer 的模式开关。
- 当前改动只保留在本地 `main`，按仓库所有者要求不要推送。

## VideoViT

- 图像实现位于 `videomamba/image_sm/models/videovit.py`，注册名为
  `videovit_tiny`、`videovit_small`、`videovit_middle`、`videovit_base`。
- 视频实现位于 `videomamba/video_sm/models/videovit.py`，注册名为
  `videovit_tiny`、`videovit_small`、`videovit_middle`。
- VideoViT block 为 pre-norm softmax attention 后接 slow SwiGLU MLP；slow MLP
  使用 `norm_mlp` 与 bias-free `gate/up/down`，默认 `mlp_ratio=3`，与 LACT
  参考实现一致。
- image/video VideoViT 的 block 参数名和运算完全一致；视频版只额外处理 3D
  tubelet embedding 与时间位置编码。
- tiny/small/middle/base 分别使用 3/6/9/12 个 attention head，head dim 固定为 64。

## VideoLACT

- 实现位于 `videomamba/video_sm/models/videolact.py`，注册名为
  `videolact_tiny`、`videolact_small`、`videolact_middle`；只在 `video_sm`
  提供，因为实验计划使用 image VideoViT 初始化 VideoViT 和 VideoLACT，
  再进行视频监督训练。
- 参考实现来自
  `/data/yibo/ttt_lvsm/configs/2026.01.05-ttt/lvsm_ttt_nanoauto_ref.yaml`
  所指向的 `model/blocks/memory_block_v2_diffusion_ref.py`。
- 此处 LACT 的 sequence mixer 是“每个 tubelet 的 window softmax attention +
  SwiGLU fast-weight memory”；两部分合起来才是 LACT，不能只保留其中之一。
  mixer 之后继续使用与 VideoViT 完全相同的 slow SwiGLU MLP。
- 一个 window/chunk 恰好对应一个嵌入后的 tubelet token，即
  `1 CLS + N spatial patches`，`window_size == chunk_size == N + 1`。每个
  tubelet 复用同一个 `cls_token` 和空间 `pos_embed`，并叠加对应的时间位置编码。
- `kernel_size/tubelet_size` 默认是 1，但允许设置为任意能整除 `num_frames` 的
  正整数。当其大于 1 时，一个 tubelet 内聚合的 token 共同完成一次
  apply-then-update，而不是逐原始帧更新。分类与回归 CLI 未显式传参时，
  `videolact_*` 默认用 1，其他已有模型仍默认用 2。
- 调度顺序必须是 apply-then-update：当前 tubelet 先依次通过所有 LACT block，
  缓存每层 memory input；整个 tubelet apply 完成后，再更新所有层的 fast
  weight；下一个 tubelet 才能使用更新后的权重。分类/回归读取最后一个
  tubelet 的 CLS。
- 每层 window attention 使用 `norm` 和 `mixer.qkv/out_proj` 命名，以便直接
  加载 image VideoViT 权重。fast-weight 分支必须使用独立的 `apply_proj`、
  `update_proj`、`value_proj`、`lr_proj` 和 `output_proj`，严禁与 window
  attention 共享 projection。
- fast weight 是多头 SwiGLU FFN 的 `w0/w1/w2`。默认 `fw_inter_multi=2`、
  `fw_num_heads=1`、`fw_base_lr=0.01`。
- 每个 tubelet 更新分别计算 `w0/w1/w2` 梯度，默认对每组梯度执行 5 次 Muon quintic
  Newton-Schulz zeroth-power 迭代，然后加到 FP32 master weight；新 fast
  weight 沿参考实现的维度归一化，并在 CUDA 上转为 BF16。
- VideoLACT 的 `norm_mlp` 和 `mlp.gate/up/down` 必须与 VideoViT 保持相同参数名、
  形状和执行位置，使 image VideoViT 的 slow MLP 权重可以直接初始化两个视频模型。

## Image ViT 初始化视频模型

- `video_sm/utils.py::adapt_image_checkpoint_for_video` 同时服务 VideoViT 和
  VideoLACT：把 2D patch kernel 中心膨胀为 3D tubelet kernel，并跳过形状
  不匹配的分类头。
- image checkpoint 没有 `temporal_pos_embedding` 时保持视频模型的零初始化；
  如果 checkpoint 中存在时间位置编码，则按目标帧数做线性插值。
- VideoLACT 从 image VideoViT 加载 patch embedding、`cls_token`、空间
  `pos_embed`、每层 window attention/norm、slow MLP、最终 norm 和形状兼容的
  head。LACT 独有的 memory、fast weight 和新增 projection 保持新初始化。
- 分类与回归入口都必须让 `videolact_*` 使用独立空间/时间位置编码的 checkpoint
  插值分支，不能误走标准 VideoMAE 的联合位置编码分支。

## 环境与兼容性

- 已测试 Conda 环境名为 `videovit`，复现文件为
  `videomamba/environment-videovit.yml`。
- 核心版本为 Python 3.11、PyTorch 2.13.0 + CUDA 13.0、torchvision
  0.28.0、timm 1.0.28、NumPy 2.4.6。
- VideoViT/VideoLACT 的图像分类、视频分类和回归路径不依赖 Mamba CUDA
  extension、Apex、DeepSpeed、TensorFlow 或 xFormers。
- 两个模型包把 `mamba_ssm` 视为可选依赖，但任何与 `mamba_ssm` 无关的导入
  错误必须继续抛出。
- 已删除 NumPy 2.x 不再提供且代码未使用的
  `numpy.lib.function_base.disp`。
- `optim_factory.py` 使用 timm 公开 optimizer export 和 `NAdam` 名称，并把
  `layers.N` 参数映射到正确的 layer-decay 深度。
- 自定义 timm factory 只显式移除 `pretrained_cfg`、
  `pretrained_cfg_overlay`、`cache_dir`。不要让模型构造函数静默吞掉任意
  `**kwargs`，否则参数拼写错误不会被发现。

## ImageNet-1K 数据规则

- 数据源使用 Hugging Face gated dataset `ILSVRC/imagenet-1k`。只下载有标签的
  train 和 validation，不下载测试集；下载前必须确保当前 Hugging Face 账号已接受
  数据集条款，token 只能从本机安全配置读取，严禁写入仓库或打印到日志。
- 原始 Parquet 固定放在
  `/mnt/localssd/dataset/videovit/imagenet-1k-hf`，展开后的训练数据固定放在
  `/mnt/localssd/dataset/videovit/imagenet-1k`。
- 本仓库的 `ImageNetDataset` 不使用 torchvision `ImageFolder` 自动推断类别，必须
  分别传入图片 root 和 `相对路径 整数标签` 格式的 meta 文件。转换脚本为
  `videomamba/image_sm/prepare_imagenet_hf.py`，支持按 shard 中断恢复并检查官方样本数。
- 训练参数固定对应为：train root `imagenet-1k/train`、train meta
  `imagenet-1k/meta/train.txt`、validation root `imagenet-1k/val`、validation meta
  `imagenet-1k/meta/val.txt`。启动完整训练前要检查 1,281,167 个 train 样本、
  50,000 个 validation 样本，并用仓库 DataLoader 实际读取样本。

## 实验输出规则

- ImageNet-1K 预训练 checkpoint 和运行日志严禁写入仓库。每个 run 使用
  `/mnt/localssd/experiments/videovit/<run-name>` 独立目录，checkpoint 写到其
  `ckpt/` 子目录，终端日志和 PID 文件写到 run 根目录。
- 本机单节点八卡训练使用 `torchrun --nnodes=1 --nproc-per-node=8`，启动前必须确认
  八张 H100 空闲。脚本应从任意工作目录都能运行，并激活 `videovit` Conda 环境。
- 当前 ImageNet-1K VideoViT-Middle 标准预训练 run 为
  `video-vit-middle-imagenet-1k-pretrain`，入口脚本是
  `videomamba/image_sm/exp/videovit_middle/run224.sh`。它使用普通 `main.py`，不使用
  teacher 或 `main_distill.py`；每卡 batch 128，八卡 global batch 1024。普通入口会
  按 global batch/512 缩放 base LR，因此 `--lr 5e-4` 对应实际 peak LR `1e-3`。
- `exp_distill/videomamba_middle` 是原作者给 VideoMamba-Middle/Base 提供的特征蒸馏
  recipe，不是 middle 模型的代码限制。除非用户明确要求，VideoViT 的标准图像预训练
  不得因为模型名是 middle 而自动切换到 distill 入口。

## Weights & Biases 规则

- image VideoViT 训练、视频分类 SFT 和视频回归 SFT 均支持 W&B；只允许主 rank
  初始化和上报，其他分布式 rank 不得创建重复 run。
- 默认 W&B entity 是 `LVSM-Experiment`，默认 project 是 `videosft`，对应页面为
  `https://wandb.ai/LVSM-Experiment/videosft`。除非用户在当次任务明确覆盖，否则
  不要修改这两个默认值。
- run name 不设自动默认值。每次启动实验时，必须根据用户对该次实验的最新指示
  显式传入 `--wandb_run_name <name>`；image 入口使用等价参数
  `--wandb-run-name <name>`。不要自行猜测、复用或覆盖 run name。
- 本机环境可能把 `WANDB_BASE_URL` 指向内网。访问公网 W&B 时，启动命令必须在
  同一个 shell 中先显式执行 `unset WANDB_BASE_URL`，再启动训练；代码初始化时也
  会再次移除该变量并打印 `unset WANDB_BASE_URL` 作为保障。
- 不要 unset `WANDB_API_KEY`，也不要在日志、命令输出、文档或提交中打印、记录
  或泄露其值。
- 标准启动形式为：先执行 `unset WANDB_BASE_URL`，再为训练命令加入 `--wandb`
  和本次指定的 run-name 参数。W&B 包已记录在 `environment-videovit.yml`。

## 已执行的验证

- VideoViT：timm registry 构造、CPU FP32 与 H100 BF16 前后向、gradient
  checkpoint、真实 `224x224` 图像和 `8x224x224` 视频推理。
- VideoViT：SDPA 与显式 `softmax(QK^T/sqrt(d))V` 数值等价。
- VideoViT：空间/时间位置编码插值后严格加载 checkpoint。
- VideoLACT：CPU apply/update 前后向，window attention、fast weight、slow MLP、
  value projection 均获得梯度；window/chunk 大小断言为单个 tubelet token 数。
- VideoLACT：调用顺序严格为每个 tubelet 先 apply 所有层、再 update 所有层；
  attention 实际输入长度始终等于 `tokens_per_tubelet`。
- VideoLACT：fast-weight 的 apply/update/value/lr/output projection 均已检查，
  与 window attention 不共享任何参数存储。
- VideoLACT：Muon NS 输出与参考实现按 BF16 运算次序逐值一致；改变较早帧并
  保持最后一帧不变会改变最终预测，确认 fast weight 正在传递跨帧信息。
- VideoLACT：GPU FP32、H100 BF16、apply/update gradient checkpoint 反向均
  通过；默认 5 次 NS 的 `videolact_tiny` 已完成 `1x3x8x224x224 -> 400`
  真实尺寸推理。
- Image ViT 初始化：VideoViT 和 VideoLACT 都已验证 2D patch kernel 中心膨胀、
  window QKV 逐值加载、类别头跳过以及 LACT-only 参数保持新初始化。
- W&B：已用 offline run 验证 image/video logger 生命周期、batch/epoch scalar
  上报、CLI 默认 entity/project、run name 必填、`WANDB_BASE_URL` 清除后选择
  `https://api.wandb.ai`，以及非主 rank 不初始化 run。
- ImageNet-1K：HF train/validation 的 308 个 Parquet shard 已完整下载并展开；文件数与
  meta 行数分别为 1,281,167 和 50,000，两个 split 都覆盖 1000 类。转换脚本已验证
  shard 级恢复，仓库真实增强和 8-worker DataLoader 已分别读取 train/validation 的
  `64x3x224x224` batch。
- 通用：AdamW/NAdam 构造、layer decay、compileall、Conda YAML dry-run 和 Git
  whitespace 检查。
