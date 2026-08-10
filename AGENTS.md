# 仓库开发说明

## 总体目标

- 本仓库用于比较 VideoMamba、VideoViT 和 VideoLACT 在图像预训练、视频监督训练下的架构能力。
- 新架构必须作为独立模型实现。不要在 VideoMamba 或 VideoViT 中加入切换 mixer 的模式开关。
- 默认不推送；只有用户在当前任务明确要求时才允许 push。

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
- tiny/small/middle/base 分别使用 3/6/9/12 个 attention head。tiny/small/base 的
  head dim 为 64；middle 使用规整的 `dim=432`、`depth=32`，head dim 为 48，
  Image/Video ViT 参数量约 78.6M。Image VideoViT、VideoViT 和 VideoLACT 的
  middle 配置必须同步，保证 image checkpoint 的共享参数形状兼容。

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
- 每个嵌入后的 tubelet 含 `1 CLS + N spatial patches`。每个 tubelet 复用同一个
  `cls_token` 和空间 `pos_embed`，并叠加对应的时间位置编码。attention window 与
  fast-weight chunk 必须覆盖相同的一组连续 tubelet：
  `window_size == chunk_size == (N + 1) * fw_update_group_size`；group size 默认 1。
- `kernel_size/tubelet_size` 默认是 1，但允许设置为任意能整除 `num_frames` 的
  正整数。当其大于 1 时，一个 tubelet 内聚合的 token 共同完成一次
  apply-then-update，而不是逐原始帧更新。分类与回归 CLI 未显式传参时，
  `videolact_*` 默认用 1，其他已有模型仍默认用 2。
- 调度依赖必须是 apply-then-update：当前 FW group 使用旧 fast weight，更新后的
  fast weight 只能供后续 group 使用。实现采用 layer-major 等价调度：每层先把
  所有独立的 FW-group-sized attention window 合并成一个 batched SDPA，再按时间
  顺序串行执行该层的 grouped memory apply/update 与 slow MLP。各层 fast state
  相互独立，因此该调度与原 group-major 依赖图等价。分类/回归读取最后一个
  tubelet 的 CLS。
- 每层 window attention 使用 `norm` 和 `mixer.qkv/out_proj` 命名，以便直接
  加载 image VideoViT 权重。默认 `share_proj=False`，fast-weight 分支保留独立
  `apply_proj`、`update_proj`、`value_proj` 和 `output_proj`；`share_proj=True`
  作为参数缩减对照，分别复用 attention Q/K/V 与 `mixer.out_proj`。
- fast weight 是多头 SwiGLU FFN 的 `w0/w1/w2`。默认 `fw_inter_multi=2`、
  `fw_num_heads=1`、`fw_base_lr=0.01`。
- 每个 FW group 更新分别计算 `w0/w1/w2` 梯度，默认对每组梯度执行 5 次 Muon quintic
  Newton-Schulz zeroth-power 迭代，然后加到 FP32 master weight；新 fast
  weight 沿参考实现的维度归一化，并在 CUDA 上转为 BF16。
- VideoLACT 的 `norm_mlp` 和 `mlp.gate/up/down` 必须与 VideoViT 保持相同参数名、
  形状和执行位置，使 image VideoViT 的 slow MLP 权重可以直接初始化两个视频模型。
- CUDA 默认在 block 级对 window attention、memory/slow-MLP apply 和 fast-weight
  update 使用 `torch.compile` 并复用 Inductor/Triton kernel。完整模型启用
  activation checkpoint 时，边界必须覆盖每层完整 recurrent scan，内部调用上述
  compiled 非 checkpoint kernel；不能只 checkpoint 单次 apply/update，否则下一
  group 仍要消费 update 输出，所有 fast/master state 历史都无法释放。整层 scan
  checkpoint 必须保留 attention dropout/drop-path RNG state；底层单算子 checkpoint
  API 继续保留用于独立调用和测试。
- stochastic depth 使用非持久化整数 tensor probability，在 kernel 内恢复 FP32，
  避免不同层的 Python `drop_prob` 导致 Dynamo 重编译，也避免 `model.to(bfloat16)`
  量化概率。共享 QKV 的 value 必须 contiguous，避免各层 stride guard 重编译。

## LACT 参考实现与 chunk/update 结论

- LVSM block 是严格 sequential residual：window attention -> fast-weight memory ->
  slow MLP；attention 与 memory 不复用 QKV，memory update 另有 `to_v/to_lr`。
  LLM recurrent LACT v4 同时支持 parallel 和 sequential residual；所检查的 124M
  配置使用 parallel、`memory_kv_mode=reuse_kv`，apply/update 复用 attention Q/K/V，
  attention 与 memory 相加后共用一个 output projection。当前 VideoLACT 保留
  sequential residual 顺序，但默认使用独立 projection。
- LVSM 参考配置为 `256/8=32`，每个 view 有 1024 tokens；训练 sample 有 12 views，
  nanoauto 实际构造 `2 * (12 - 1)=22` 个 image-token block，总 sequence length 为
  22,528。`parallel_ttt_config` 交替执行 1024-token update block 和 1024-token
  apply-only block，因此 `seqlen / update_chunk = 22`，每层每个 sample 实际更新
  11 次，平均每 2048 tokens 更新一次。
- LVSM 的 transformer 对照不是一次 full-sequence attention 的标准 ViT，而是
  recurrent KV-cache transformer：同样串行处理上述 22 个 1024-token op，每个
  op 都逐层调用 attention；11 个 update op 才把新 K/V 写入持久 cache。配置也不是
  同 batch：transformer 每卡 16，LACT 每卡 8。因此相近 step time 不代表相近
  sample throughput。直接调用两个 depth-8 block stack、B1/22,528-token BF16
  前后向时，transformer 为约 423 ms/1.51 GiB，LACT 为 563 ms/1.87 GiB；按配置
  batch 16/8 测得约 938/725 ms，对应 17.1/11.0 samples/s。transformer 环境缺少
  `flash_attn` 包时用等价 PyTorch SDPA H100 flash backend 完成此对照。
- 加入 `transformer_vgap8_ablation5.yaml` 后，使用三个配置各自 batch 对完整
  `Images2latent3D` 做合成 12-view 256² BF16 前后向（含 tokenizer/decoder/loss，
  不含 optimizer step、FSDP 和 dataloader）：vgap-null Transformer B16 为
  940.6 ms/22.84 GiB/17.01 samples/s；vgap8 per-layer-KV Transformer B16 为
  942.1 ms/20.44 GiB/16.98 samples/s；LACT reference B8 为
  728.7 ms/14.59 GiB/10.98 samples/s。按配置看 LACT 每步总峰值显存反而低
  36%/29%，但折算每 sample 为 1.82 GiB，分别比两个 Transformer 的
  1.43/1.28 GiB 高约 28%/43%；较小 batch 掩盖了单样本内存和吞吐成本。
- LLM recurrent LACT v4 的训练 sequence length 为 32,768，window/chunk 都是
  4,096，共 8 个 chunk；实现跳过最后一个 chunk 的 memory update，因此每层每个
  sample 实际更新 7 次，`seqlen / chunk = 8`。
- 当前 VideoLACT 在 `224/16` 下每个嵌入 tubelet 为 `1 + 14 * 14 = 197` tokens。
  `fw_update_group_size` 会把多个连续 tubelet 同时合并为一个 attention window 和
  一次 FW apply/update chunk：4/5/6/8 个 tubelet 分别对应
  788/985/1182/1576 tokens，20/21 个对应 3940/4137 tokens。
- 若每个 FW group 都 update 且跳过最后一次无后继状态的 update，设视频有 `F`
  帧、temporal tubelet size 为 `k`、每个 FW group 含 `g` 个嵌入 tubelet，则
  `num_groups = ceil((F / k) / g)`，对最终预测有效的 update 数为
  `num_groups - 1`。建议首轮采用 48 帧、`k=1`、`g=6`：约 1182 tokens/chunk、
  8 groups、7 次有效 update，update 次数与 LLM 一致且 chunk 大小接近 LVSM。
  若希望接近 LVSM 的 11 次 update，可用 60 帧/`g=5` 或 72 帧/`g=6`。若同时
  追求 LLM 式约 4K chunk 和 7 次 update，则需要约 160 帧/`g=20` 或
  168 帧/`g=21`，成本过高，不建议作为首轮实验。
- 实现会跳过最后一个 FW group 的无效 update，因为产生的 state 没有后续 token
  消费。64 帧、`tubelet_size=1`、`fw_update_group_size=8` 时，每层有 8 个
  1576-token group，实际 update 7 次。
- fast-weight 参数大头来自 `w0/w1/w2`。当前 Middle 为 `dim=432`、depth 32、
  1 FW head、`fw_inter_multi=2`：每层 1,119,744、全模型 35,831,808 个 FW 参数。
  LVSM 为 `dim=512`、depth 8、1 head、inter 2：每层 1,572,864、总计
  12,582,912。LLM 124M 为 `dim=768`、depth 12、4 heads、per-head inter 1：
  runtime FW 每层 442,368、总计 5,308,416；w0/w2 使用 rank-32 low-rank 参数化。
- 若保持 `dim=432`、默认 private projection 与其他 block 配置不变，VideoLACT
  depth 18 在 64 帧/400 类下为 77,959,120 参数，与 depth-32 VideoViT 的
  78,337,552 参数只差 -0.48%，可作为参数匹配补充对照。若沿 LVSM protocol 使用
  同深度主对照，必须同时报告参数量、显存和吞吐，不能称为 parameter-matched。
- 同为 `dim=512`、1 FW head、inter 2、sequence `16 * 1024`、chunk 1024、
  5-step Muon 的单层 H100 BF16 对照中，本仓库 shared-proj eager 为 163.12 ms，
  compiled 为 62.72 ms，compiled+checkpoint 为 71.42 ms；LVSM compiled+checkpoint
  为 78.81 ms。实现本身与 LVSM 同量级；未分组的 197-token update 会过于频繁，
  而分组后 B1 latency 仍会因小矩阵 GPU 利用率过低而夸大，训练吞吐应在能占满 GPU
  的 batch 下比较。

## Image ViT 初始化视频模型

- `video_sm/utils.py::adapt_image_checkpoint_for_video` 同时服务 VideoViT 和
  VideoLACT：把 2D patch kernel 中心膨胀为 3D tubelet kernel，并跳过形状
  不匹配的分类头。
- image checkpoint 没有 `temporal_pos_embedding` 时保持视频模型的零初始化；
  如果 checkpoint 中存在时间位置编码，则按目标帧数做线性插值。
- VideoLACT 从 image VideoViT 加载 patch embedding、`cls_token`、空间
  `pos_embed`、每层 window attention/norm、slow MLP、最终 norm 和形状兼容的
  head。LACT 独有的 memory、fast weight、`lr_proj` 以及 private 模式下的新增
  projection 保持新初始化。
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
- 当前 PyTorch 默认使用 `torch.load(weights_only=True)`；`video_sm` 的监督分类和
  回归 finetune/resume checkpoint 包含 `argparse.Namespace` 与 optimizer 状态，
  因此可信的本仓库 checkpoint 路径必须显式传入 `weights_only=False`。

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

## Kinetics-400 数据规则

- K400 视频使用 Hugging Face `9tlofrjjlcq5/k400`，CSV 使用 VideoMamba 官方
  `k400.zip` 中的数据列表；不要使用 HF 仓库里混合 K400/K700 的
  `k710_train.csv`。原始视频归档保存在 `/mnt/localssd/dataset/videovit/k400-hf`。
- 展开后 metadata root 为 `/mnt/localssd/dataset/videovit/k400/kinetics_400`，
  video root 为其 `videos_320/`；训练对应传入前者作为 `--data_path`、后者作为
  `--prefix`，并使用 `--data_set Kinetics_sparse --split ',' --nb_classes 400`。
- 当前共有 240,436 个 train、19,787 个 validation；`test.csv` 与 `val.csv`
  相同。所有 split 都覆盖 400 类且没有缺失视频；目录共有 260,232 个 MP4，其中
  9 个未被 train/val split 使用。
- 下载分片 SHA-256、ZIP CRC、官方 CSV 文件覆盖，以及仓库 Decord 4-worker
  DataLoader 的 train/validation `4x3x16x224x224` batch 均已验证通过。

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
- 该 run 最初误用 `dim=576` 启动出 139.3M ViT；发现后已在首个 epoch 完成前停止，
  不得从该尝试恢复。正式 run 必须使用 `dim=432` 的约 78.6M 模型并从头训练。
- `exp_distill/videomamba_middle` 是原作者给 VideoMamba-Middle/Base 提供的特征蒸馏
  recipe，不是 middle 模型的代码限制。除非用户明确要求，VideoViT 的标准图像预训练
  不得因为模型名是 middle 而自动切换到 distill 入口。
- K400 VideoViT-Middle F8 监督训练的 run name 为
  `video-vit-middle-k400-f8x224`，入口脚本为
  `videomamba/video_sm/exp/k400/videovit_middle/run_f8x224.sh`，输出固定放在
  `/mnt/localssd/experiments/videovit/<run-name>/ckpt`。使用 ImageNet run 的
  `best_checkpoint.pth` 初始化，数据使用 `kinetics_400` metadata root 和
  `videos_320/` prefix。
- K400 F8 超参数对齐原始 `videomamba_middle/run_f8x224.sh`：50 epochs、每卡
  batch 32、`num_sample=2`、tubelet size 1、base LR `2e-4`、warmup 5 epochs、
  drop path 0.8、layer decay 0.75。原 recipe 的 16 卡 global batch 512 在本机
  8 卡上通过 `update_freq=2` 保持不变；入口按 repeated sample 继续缩放，实际
  peak LR 为 `8e-4`。
- 简要运行方式：确保 8 张 GPU 空闲且当前 shell 已提供 `WANDB_API_KEY`，执行
  `unset WANDB_BASE_URL` 后运行
  `bash videomamba/video_sm/exp/k400/videovit_middle/run_f8x224.sh`。脚本会自行激活
  `videovit` 环境；后台运行时把 stdout/stderr 写入对应 run 目录的 `train.log`，
  并把 launcher PID 写入 `train.pid`。
- 首次 K400 F8 run 最终 12-view test 为 top-1 73.154%、top-5 91.257%，最佳
  single-view validation top-1 为 71.433%。该 run 使用 `layer_decay=0.75`；在 32 层
  模型上底层 peak LR 只有约 `6e-8`，checkpoint 中 layer 0 QKV 相对 ImageNet
  初始化仅变化约 0.21%。
- no-layer-decay 对照 run 为 `video-vit-middle-k400-f8x224-no-ld`，入口脚本是
  `videomamba/video_sm/exp/k400/videovit_middle/run_f8x224_no_ld.sh`。它只把
  `--layer_decay` 改为 `1.0` 并使用独立输出/W&B run name，其他训练和评估参数与
  首次 F8 run 完全相同。`layer_decay=1.0` 会关闭 `LayerDecayValueAssigner`，所有层
  使用相同的 scheduler LR。ImageNet 预训练入口不使用 `video_sm` 的这套 layer-wise
  LR decay。
- K400 VideoLACT-Middle F8 no-layer-decay run 为
  `video-lact-middle-k400-f8x224-no-ld`，入口脚本是
  `videomamba/video_sm/exp/k400/videolact_middle/run_f8x224_no_ld.sh`，公网 W&B run
  为 `https://wandb.ai/LVSM-Experiment/videosft/runs/da1r720a`。除模型外沿用对应
  VideoViT F8 no-LD recipe：每卡 `batch_size=32`、`update_freq=2`、
  `num_sample=2`、8 帧、base LR `2e-4`、layer decay 1.0；LACT 使用 private
  projection、`fw_update_group_size=1`，即每个 sampled frame 是一个 197-token
  attention/FW group，共 8 group/7 次有效 state update。真实 B64 不使用 checkpoint
  时首个 forward 在约 79.15 GiB OOM，因此该脚本必须启用 32 层完整 scan
  checkpoint；这是与 ViT recipe 的执行差异。单卡完整 AdamW step 验证为约
  5.70 s/33.64 GiB；正式八卡 run 于 2026-08-10 实测稳态约
  6.07 s/micro-step，PyTorch peak 35.59 GiB、`nvidia-smi` 约 39.03 GiB/卡，因
  训练耗时不可接受在约 32 分钟后主动停止。所有 rank/GPU 已释放，未产生 epoch
  checkpoint，不得自动恢复该次尝试。
- K400 F64 no-layer-decay VideoViT/VideoLACT recipe 已分别准备在
  `videomamba/video_sm/exp/k400/videovit_middle/run_f64x224_no_ld.sh` 和
  `videomamba/video_sm/exp/k400/videolact_middle/run_f64x224_no_ld.sh`，run name 为
  `video-vit-middle-k400-f64x224-no-ld-2ep` 与
  `video-lact-middle-k400-f64x224-no-ld-2ep`。两者沿原 VideoMamba F64 recipe 使用
  每卡 `batch_size=4`、`num_sample=2`、64 帧、224²、tubelet size 1、drop path 0.8，
  并显式设置 `layer_decay=1.0` 和 32 层完整 activation checkpoint；12 小时快速对照
  将 schedule 压缩为 2 epochs/15,026 optimizer steps，其中 1,500 steps warmup，
  CLI `lr/warmup_lr/min_lr=4e-4/4e-6/4e-6`，按当前 global batch 缩放后的实际值为
  `1e-4/1e-6/1e-6`。LACT 额外使用 private projection、
  `fw_update_group_size=8`，即 8 个 1576-token attention/FW group。VideoViT 曾于
  2026-08-10 以旧 run name
  `video-vit-middle-k400-f64x224-no-ld` 启动，公网 W&B run 为
  `https://wandb.ai/LVSM-Experiment/videosft/runs/fxgm8m92`；实测每 epoch 训练约
  2 小时 54 分、完整 50 epochs ETA 约 6.4 天，因耗时不可接受已在 epoch 0
  step 397/7513 主动停止。所有 rank/GPU 已释放，未生成 epoch checkpoint，不得
  自动恢复该次尝试。2-epoch VideoViT 已于 2026-08-10 启动，公网 W&B run 为
  `https://wandb.ai/LVSM-Experiment/videosft/runs/we1s81t4`，tmux session/PID 为
  `video-vit-middle-k400-f64x224-no-ld-2ep`/`2388158`；稳定 step time 约 1.40 秒，
  PyTorch peak 9.20 GiB、`nvidia-smi` 约 14.04 GiB/卡。2-epoch LACT 尚未 launch。

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
  会再次移除该变量并打印 `unset WANDB_BASE_URL`，且向 `wandb.init` 显式传入
  `https://api.wandb.ai` settings，避免用户级 W&B settings 重新选择内网 endpoint。
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
  private/shared projection 与 `lr_proj` 均获得梯度；group size 1/2 的前后向以及
  `chunk_size == window_size == tokens_per_tubelet * fw_update_group_size` 已验证。
- VideoLACT：layer-major 调度与原 tubelet-major apply-then-update 实现已做输出和
  梯度等价检查；每层所有完整 grouped window 合并为一次
  `(B*num_groups, group_size*197, D)` SDPA，fast-weight apply/update 仍按 group
  时间顺序串行。非整除的最后一个短 group 使用独立形状的 SDPA。
- VideoLACT：默认 private projection 路径及 shared projection 对照路径均已验证；
  shared 模式下 Q/K/V 分别进入 FW apply/update/target，memory output 复用
  attention out projection。
- VideoLACT：Muon NS 输出与参考实现按 BF16 运算次序逐值一致；改变较早帧并
  保持最后一帧不变会改变最终预测，确认 fast weight 正在传递跨帧信息。
- VideoLACT：GPU FP32、H100 BF16、apply/update gradient checkpoint 反向均
  通过；默认 5 次 NS 的 `videolact_tiny` 已完成 `1x3x8x224x224 -> 400`
  真实尺寸推理。
- Image ViT 初始化：VideoViT 和 VideoLACT 都已验证 2D patch kernel 中心膨胀、
  window QKV 逐值加载、类别头跳过以及 LACT-only 参数保持新初始化。
- VideoLACT：默认 CUDA compiled 和 compiled+checkpoint 路径均已在 H100 BF16
  验证，并确认生成 fused RMSNorm、SwiGLU/softplus backward、Muon bmm+norm 等
  Triton kernel；整层 recurrent-scan checkpoint 已在 CPU private/shared 模式以及
  H100 BF16 compiled private 模式验证，带 attention dropout/drop-path 时输出、输入
  梯度和所有参数梯度均与 eager 路径逐值一致（最大差值 0）。
- K400 Middle H100 BF16 前后向对照（224、400 类、drop-path 0.8、FP32 参数与梯度、
  无 optimizer/FSDP/dataloader、全层 checkpoint、编译预热不计）使用脚本的真实
  单卡模型 batch：F32 脚本 `batch_size=8,num_sample=2`，collate 后 B16；F64 脚本
  `batch_size=4,num_sample=2`，collate 后 B8。两者均为 8 个 FW group，F32 每组
  4 帧/788-token attention window，F64 每组 8 帧/1576-token window；最后一个
  无后继 state 的 update 仍跳过，因此实际写入 7 次。稳定复测结果：F32 的
  ViT/LACT 为 839/2527 ms、19.07/6.33 clips/s、8.32/16.86 GiB，LACT 是
  3.01 倍 step time、2.03 倍显存；F64 为 1315/2181 ms、6.08/3.67 clips/s、
  8.32/14.16 GiB，LACT 是 1.66 倍 step time、1.70 倍显存。ViT 分别对
  6,273/12,545 tokens 做 full attention。
- F64 launch-setting 单 H100 验证进一步加载了正式 ImageViT checkpoint，并把
  label-smoothed loss、AdamW step 以及初始化后的一阶/二阶 optimizer state 计入
  峰值；脚本 `batch_size=4,num_sample=2` 在 collate 后实际进入模型 B8。VideoViT
  为 1314 ms/6.09 clips/s/8.91 GiB，VideoLACT 为
  2258 ms/3.54 clips/s/15.21 GiB。两者均连续完成 5 个计时训练步，距 80 GiB
  单卡容量有充足余量；该测试不含 DDP bucket、视频解码和 dataloader。
- 改成 grouped attention window 之前，F64 private-proj LACT 的 197-token
  per-tubelet window 在 B8 为约 2071 ms/14.16 GiB；扩大为 1576-token window 后
  为约 2181 ms/14.16 GiB，即显存基本不变、step time 增加约 5%。旧逐 update
  checkpoint 的 F64 B16 峰值为 56.34 GiB；这是 checkpoint 粒度问题，不能与当前
  grouped-window B8 结果直接比较。
- F64 LACT B16 显存归因：depth 32 为 56.33 GiB，完全相同配置改成 depth 8 仅
  15.90 GiB；`residual_in_fp32` true/false 均为 15.90 GiB，当前峰值不是该开关
  导致。depth-32 baseline 在 forward 结束时已常驻 55.16 GiB，backward 只把峰值
  增至 56.33 GiB，说明逐 update checkpoint 仍保留了 32 层的 fast/master state
  历史。现已正式把 checkpoint 边界扩大到每层完整 8-group recurrent scan，峰值
  降到 27.76 GiB（-50.7%）；稳定复测 step 约 3.15 s，相比旧路径约 2.81 s 慢
  约 12%。`use_checkpoint/checkpoint_num` 现在控制整层 scan checkpoint。
- Middle 缩放：`dim=432`、`depth=32`、9 heads 配置下，Image ViT 为
  78,569,704 参数，VideoViT 为 78,573,160 参数；默认 private-proj VideoLACT 为
  138,348,136 参数，shared-proj 对照为 114,460,264 参数。image checkpoint 与
  两个视频模型的共享参数形状已逐项验证兼容。
- W&B：已用 offline run 验证 image/video logger 生命周期、batch/epoch scalar
  上报、CLI 默认 entity/project、run name 必填、`WANDB_BASE_URL` 清除后选择
  `https://api.wandb.ai`，以及非主 rank 不初始化 run。
- ImageNet-1K：HF train/validation 的 308 个 Parquet shard 已完整下载并展开；文件数与
  meta 行数分别为 1,281,167 和 50,000，两个 split 都覆盖 1000 类。转换脚本已验证
  shard 级恢复，仓库真实增强和 8-worker DataLoader 已分别读取 train/validation 的
  `64x3x224x224` batch。
- K400 VideoViT-Middle F8：ImageNet best checkpoint 初始化加载了 78,136,704 个
  trunk 参数，仅时间位置编码和 400 类 head 新初始化；真实 K400 repeated sample
  解码为两份 `3x8x224x224` clip。单张 H100 上用实际每 microbatch 64 clips、
  累积两次并执行 AdamW step 已通过，峰值显存为 61.54 GiB。
- 通用：AdamW/NAdam 构造、layer decay、compileall、Conda YAML dry-run 和 Git
  whitespace 检查。
