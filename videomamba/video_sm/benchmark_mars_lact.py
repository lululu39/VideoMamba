"""Single-H100 synthetic training-step benchmark for VideoMARS/VideoLACT."""

import argparse
import gc
import statistics
import time

import torch
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model

from models import *  # noqa: F403 - import registrations used by timm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("videomars_middle", "videolact_middle"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--fw-update-group-size", type=int, default=8)
    parser.add_argument("--lact-layer-group-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--memory-gate", type=float, default=0.1)
    parser.add_argument("--checkpoint-num", type=int, default=32)
    parser.add_argument("--mars-encoder-dim", type=int, default=None)
    parser.add_argument("--mars-encoder-depth", type=int, default=1)
    parser.add_argument("--mars-encoder-num-heads", type=int, default=None)
    parser.add_argument("--mars-decoder-dim", type=int, default=32)
    parser.add_argument("--mars-decoder-depth", type=int, default=1)
    parser.add_argument("--mars-decoder-num-heads", type=int, default=1)
    return parser.parse_args()


def make_model(args):
    common = dict(
        img_size=224,
        pretrained=False,
        num_classes=400,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.8,
        mlp_ratio=3.0,
        kernel_size=1,
        num_frames=args.num_frames,
        fw_inter_multi=2,
        fw_update_group_size=args.fw_update_group_size,
        muon_update_steps=5,
        use_checkpoint=args.checkpoint_num > 0,
        checkpoint_num=args.checkpoint_num,
    )
    if args.model == "videolact_middle":
        common.update(
            fw_num_heads=1,
            fw_base_lr=0.01,
            fw_update_layer_group_size=args.lact_layer_group_size,
            share_proj=False,
            share_init=True,
        )
    else:
        common.update(
            mars_encoder_dim=args.mars_encoder_dim,
            mars_encoder_depth=args.mars_encoder_depth,
            mars_encoder_num_heads=args.mars_encoder_num_heads,
            mars_decoder_dim=args.mars_decoder_dim,
            mars_decoder_depth=args.mars_decoder_depth,
            mars_decoder_num_heads=args.mars_decoder_num_heads,
        )
    model = create_model(args.model, **common)
    with torch.no_grad():
        if getattr(model, "shared_state", None) is not None:
            model.shared_state.memory_gate.fill_(args.memory_gate)
        for layer in model.layers:
            if getattr(layer, "memory_gate", None) is not None:
                layer.memory_gate.fill_(args.memory_gate)
    return model


def run_step(model, optimizer, criterion, samples, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(samples)
        loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    gc.disable()

    device = torch.device("cuda", 0)
    model = make_model(args).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    samples = torch.randn(
        args.batch_size,
        3,
        args.num_frames,
        224,
        224,
        device=device,
        dtype=torch.bfloat16,
    )
    targets = torch.randint(0, 400, (args.batch_size,), device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"CONFIG model={args.model} batch={args.batch_size} "
        f"frames={args.num_frames} group={args.fw_update_group_size} "
        f"parameters={parameter_count}",
        flush=True,
    )

    try:
        for step in range(args.warmup_steps):
            start = time.perf_counter()
            loss = run_step(model, optimizer, criterion, samples, targets)
            torch.cuda.synchronize()
            print(
                f"WARMUP step={step} seconds={time.perf_counter() - start:.6f} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for step in range(args.measure_steps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            loss = run_step(model, optimizer, criterion, samples, targets)
            end_event.record()
            end_event.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
            timings.append(elapsed)
            print(
                f"MEASURE step={step} seconds={elapsed:.6f} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        median = statistics.median(timings)
        print(
            f"RESULT model={args.model} batch={args.batch_size} "
            f"median_seconds={median:.6f} clips_per_second={args.batch_size / median:.6f} "
            f"peak_allocated_gib={peak_allocated:.6f} "
            f"peak_reserved_gib={peak_reserved:.6f}",
            flush=True,
        )
    except torch.OutOfMemoryError as error:
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        print(
            f"OOM model={args.model} batch={args.batch_size} "
            f"peak_allocated_gib={peak_allocated:.6f} "
            f"peak_reserved_gib={peak_reserved:.6f} error={error}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
