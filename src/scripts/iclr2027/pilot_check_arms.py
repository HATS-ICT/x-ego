#!/usr/bin/env python3
"""
P0 completion gate -- inspect the actual arms before the full launch.

Checks, on train/val data only:
  * the instantiated model has NO temporal block and the expected trainable
    parameter count (3 final backbone blocks),
  * positive-pair masks are what each arm claims (group structure, and for C the
    members really are distinct same-time teammates),
  * frame aggregation is plain averaging over 20 frames,
  * unique-clip vs processed-view budgets per arm (the C-S exposure caveat),
  * measured steps/second on a small number of batches, for the GPU-hour estimate.

Usage:
  uv run python -m src.scripts.iclr2027.pilot_check_arms --arms cecl single shuffle
  uv run python -m src.scripts.iclr2027.pilot_check_arms --arms cecl --timed-batches 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich import print as rprint

from main import setup_base_pathing
from src.data_module.contrastive import ContrastiveDataModule
from src.models.contrastive_model import ContrastiveModel
from src.utils.config_utils import load_cfg, apply_cfg_overrides

CONFIG = "configs/train/contrastive_iclr2027.yaml"
SHUFFLE_MANIFEST = "contrastive_matchheldout_v1_shuffle_s{seed}.csv"


def build_cfg(arm: str, seed: int):
    cfg = load_cfg(CONFIG)
    overrides = [f"data.pair_mode={arm}", f"meta.seed={seed}"]
    if arm == "shuffle":
        overrides.append(f"data.shuffle_manifest={SHUFFLE_MANIFEST.format(seed=seed)}")
    cfg = apply_cfg_overrides(cfg, overrides)
    cfg = setup_base_pathing(cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"path": {"exp": "/tmp"}}))
    return cfg


def inspect_batches(dm, arm: str, n_batches: int) -> dict:
    loader = dm.train_dataloader()
    model_free_stats = {
        "batches": 0,
        "videos": 0,
        "group_sizes": Counter(),
        "unique_clips": set(),
        "processed_views": 0,
        "positive_pairs": 0,
        "distinct_agents_in_group": Counter(),
    }
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        counts = batch["agent_counts"].tolist()
        total = int(batch["video"].shape[0])
        model_free_stats["batches"] += 1
        model_free_stats["videos"] += total
        model_free_stats["processed_views"] += total
        for c in counts:
            model_free_stats["group_sizes"][c] += 1
            # positives within a group, excluding the diagonal
            model_free_stats["positive_pairs"] += c * (c - 1)

        offset = 0
        for c in counts:
            agents = batch["agent_ids"][offset : offset + c]
            model_free_stats["distinct_agents_in_group"][len(set(agents))] += 1
            offset += c
        model_free_stats["unique_clips"].update(batch["agent_ids"])

    out = dict(model_free_stats)
    out["group_sizes"] = dict(sorted(out["group_sizes"].items()))
    out["distinct_agents_in_group"] = dict(sorted(out["distinct_agents_in_group"].items()))
    out["unique_agent_ids_seen"] = len(out.pop("unique_clips"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot audit of the P1 arms")
    parser.add_argument("--arms", nargs="+", default=["cecl", "single", "shuffle"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--timed-batches", type=int, default=0,
                        help="if > 0, also run this many encoder forward passes for a steps/s estimate")
    args = parser.parse_args()

    report = {"seed": args.seed, "arms": {}}

    for arm in args.arms:
        rprint(f"\n[bold cyan]=== arm: {arm} ===[/bold cyan]")
        cfg = build_cfg(arm, args.seed)

        dm = ContrastiveDataModule(cfg)
        dm.prepare_data()
        dm.setup("fit")

        arm_report = {
            "train_groups": len(dm.train_dataset),
            "val_groups": len(dm.val_dataset),
        }
        arm_report["batch_audit"] = inspect_batches(dm, arm, args.batches)

        # Model-side checks: no temporal block, expected trainable parameters.
        # VideoEncoder is a factory wrapper; the concrete encoder (which owns the
        # optional temporal transformer) lives at .video_encoder.video_encoder.
        model = ContrastiveModel(cfg)
        encoder = model.video_encoder.video_encoder
        arm_report["encoder_class"] = type(encoder).__name__
        if not hasattr(encoder, "temporal"):
            raise AssertionError(
                f"{type(encoder).__name__} has no 'temporal' attribute; the pilot cannot "
                f"verify frame averaging. Fix the introspection path before launching."
            )
        temporal = encoder.temporal
        arm_report["temporal_block"] = "None" if temporal is None else type(temporal).__name__
        if temporal is not None:
            raise AssertionError(
                f"temporal block is {type(temporal).__name__}, expected None. The P1 protocol "
                f"requires plain frame averaging (model.encoder.temporal_heads: null)."
            )
        arm_report["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        arm_report["total_params"] = sum(p.numel() for p in model.parameters())

        # Frame aggregation: a constant-in-time input must survive mean pooling.
        with torch.no_grad():
            probe = torch.zeros(1, cfg.data.fixed_duration_seconds * cfg.data.target_fps, 3, 224, 224)
            feats = encoder(probe)
        arm_report["encoder_output_shape"] = list(feats.shape)
        arm_report["frames_per_clip"] = int(probe.shape[1])

        if args.timed_batches > 0 and torch.cuda.is_available():
            # The real training loop runs each microbatch TWICE: a no_grad pass to
            # fill the embedding cache, then a replay pass with backward. Timing a
            # single plain forward/backward would understate the per-step cost, so
            # both passes are measured. `warmup` batches are excluded so cold page
            # cache and dataloader spin-up do not pollute the rate.
            model = model.cuda().train()
            loader = dm.train_dataloader()
            precision = str(cfg.training.precision)
            if "bf16" not in precision:
                raise AssertionError(
                    f"pilot timing assumes bf16 autocast, but training.precision={precision}"
                )
            autocast_dtype = torch.bfloat16
            warmup = 5
            t_cache = t_replay = 0.0
            done = 0
            for i, batch in enumerate(loader):
                batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                # Lightning applies precision=bf16-mixed via autocast inside the
                # Trainer. Calling the model directly would run fp32 and overstate
                # the per-microbatch cost several-fold, so autocast is applied here
                # to match what the real run actually executes.
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
                    model._compute_projected_embeddings(batch)
                torch.cuda.synchronize(); t1 = time.time()

                with torch.autocast("cuda", dtype=autocast_dtype):
                    out = model.forward(batch)
                out["loss"].backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); t2 = time.time()

                if i >= warmup:
                    t_cache += t1 - t0
                    t_replay += t2 - t1
                    done += 1
                if done >= args.timed_batches:
                    break

            per_microbatch = (t_cache + t_replay) / max(done, 1)
            accum = cfg.training.contrastive_accumulate_batches
            micro_per_epoch = int(dm.train_dataset.df["num_alive_teammates"].sum()) / cfg.data.batch_size
            arm_report["timing"] = {
                "timed_microbatches": done,
                "sec_per_microbatch": round(per_microbatch, 3),
                "microbatches_per_sec": round(1.0 / per_microbatch, 3) if per_microbatch else None,
                "sec_cache_pass": round(t_cache / max(done, 1), 3),
                "sec_replay_pass": round(t_replay / max(done, 1), 3),
                "sec_per_optimizer_step": round(per_microbatch * accum, 2),
                "microbatches_per_epoch": round(micro_per_epoch, 1),
                "est_hours_per_epoch": round(micro_per_epoch * per_microbatch / 3600, 2),
                "est_hours_40_epochs": round(micro_per_epoch * per_microbatch * 40 / 3600, 1),
                "precision": precision,
                "note": (
                    "both passes of the embedding-cache loop (no_grad cache + replay "
                    "forward/backward) under bf16 autocast; excludes dataloader wait"
                ),
            }

        report["arms"][arm] = arm_report
        rprint(json.dumps(arm_report, indent=2, default=str))
        del model

    out_path = Path("artifact") / f"iclr2027_pilot_s{args.seed}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    rprint(f"\n[green]OK[/green] pilot report: [bold]{out_path}[/bold]")


if __name__ == "__main__":
    main()
