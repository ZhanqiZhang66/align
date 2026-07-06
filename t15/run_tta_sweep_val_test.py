#!/usr/bin/env python3
"""
TTA sweep: ours (MDAN) + GRU. Val and test are always cumulative (no independent per-day runs).
Saves one JSON per seed. Each run stores:
  - Val: cumulative TTA over val days; per-day WER along that pass; val-init ckpt at the end.
  - Test (no val init): cumulative TTA over test days from original ckpt; per-day WER.
  - Test (val init): cumulative TTA over test days from val-init ckpt; per-day WER.
Load all seed JSONs and plot mean ± std. Default 3 seeds. GRU TTA lr=0.02, ours 0.03.
Use --multi-gpu to run one seed per GPU in subprocesses (avoids OOM when LM decoder fills cuda:0).

tmux new -s tta_sweep 'cd /victoriapvc && export LM_DECODER_DIR=/victoriapvc/data/willett/lm/languageModel && export BRAIN2TEXT_LM_DIR=/victoriapvc/data/willett/lm/languageModel && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && PYTHONPATH=/victoriapvc/repos/brain2text-t15:/victoriapvc/pip_packages python repos/brain2text-t15/run_tta_sweep_val_test.py --split split1 --multi-gpu --num-seeds 3 --seed0 123 --ours-tta-lrs 0.03 --gru-tta-lr 0.02 --results-dir /victoriapvc/results/wer_t15_new_tta; echo "Exit: $?"; read -p "Press Enter to close"'
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup (match notebook)
# ---------------------------------------------------------------------------
BASE = os.environ.get("BASE", "/victoriapvc")
REPO = os.path.join(BASE, "repos", "brain2text-t15")
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(BASE, "pip_packages"))
os.environ.setdefault("LM_DECODER_DIR", "/victoriapvc/data/willett/lm/languageModel")
os.environ.setdefault("BRAIN2TEXT_LM_DIR", "/victoriapvc/data/willett/lm/languageModel")

import run_n_gram_wer_eval as ev
from tta_utils import clean_transcription, get_phonemes
from lm_utils import _cer_and_wer


def transcription_tensor_to_str(t):
    a = t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if (a == 0).any():
        a = a[: int(np.where(a == 0)[0][0])]
    return bytes(a.tolist()).decode("utf-8", errors="ignore")


def freeze_linear_paramlist_like_tensor(model):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("day_weights.") or name.startswith("day_biases."):
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No day_weights/day_biases params found.")
    return trainable


def lm_decode_known_good(logits_btC):
    T = int(logits_btC.shape[1])
    decoded, _, _ = ev.get_lm_outputs(logits_btC, n_frames=T)
    return clean_transcription(decoded)


def run_gru_tta_day_only(
    model,
    forced_day,
    loader,
    lr,
    tta_args,
    device,
    *,
    debug_first_n=0,
    score_on="post",
):
    """TTA on one day's loader. Uses state_dict save/restore for PRE (no frozen copy) to save GPU memory."""
    assert score_on in {"pre", "post"}
    n_augs = int(tta_args["repeats"][0])
    adapt_steps = int(tta_args["adaptation_steps"])
    use_aug = bool(tta_args.get("WN+BS", False))
    wn = float(tta_args.get("white_noise", 0.0))
    bs = float(tta_args.get("baseline_shift", 0.0))
    wd = float(tta_args.get("l2_decay", 0.0))

    if isinstance(forced_day, torch.Tensor):
        forced_day_t = forced_day.detach().to(device=device, dtype=torch.long).view(-1)
        forced_day_t = forced_day_t[:1] if forced_day_t.numel() else torch.tensor([0], device=device, dtype=torch.long)
    else:
        forced_day_t = torch.tensor([int(forced_day)], device=device, dtype=torch.long)

    trainable_params = freeze_linear_paramlist_like_tensor(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd, eps=0.1, betas=(0.9, 0.999))
    # Keep initial state on CPU to avoid 2x model on GPU
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    decoded_list, ref_list = [], []
    for batch_idx, batch in enumerate(loader):
        features = batch["input_features"].to(device)
        B = int(features.shape[0])
        day_idx = forced_day_t.expand(B)
        ref_raw = transcription_tensor_to_str(batch["transcriptions"][0])
        ref = clean_transcription(ref_raw)

        # PRE from initial state (no frozen model copy)
        current_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in initial_state.items()}, strict=True)
        model.eval()
        with torch.no_grad():
            logits_pre = model(features, day_idx)
            decoded_pre = lm_decode_known_good(logits_pre)
        model.load_state_dict({k: v.to(device) for k, v in current_state.items()}, strict=True)

        y_pseudo, y_len_pseudo = get_phonemes(decoded_pre)
        y_pseudo = y_pseudo.to(device)
        y_len_pseudo = (
            y_len_pseudo.to(device)
            if isinstance(y_len_pseudo, torch.Tensor)
            else torch.tensor([int(y_len_pseudo)], device=device)
        )

        if adapt_steps > 0:
            if n_augs > 1:
                X = features.repeat(n_augs, 1, 1)
                day_rep = forced_day_t.expand(X.shape[0])
                y_rep = y_pseudo.unsqueeze(0).repeat(n_augs, 1)
                ylen_rep = y_len_pseudo.repeat(n_augs)
            else:
                X = features
                day_rep = day_idx
                y_rep = y_pseudo.unsqueeze(0)
                ylen_rep = y_len_pseudo
            if use_aug and n_augs > 1:
                X = X + torch.randn_like(X) * wn
                X = X + torch.randn((X.shape[0], 1, X.shape[2]), device=device) * bs
            model.train()
            for _ in range(adapt_steps):
                logits = model(X, day_rep)
                T = int(logits.shape[1])
                lens = torch.full((logits.shape[0],), T, device=device, dtype=torch.int32)
                loss = ev._ctc_loss_exact(logits, y_rep, lens, ylen_rep)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_post = model(features, day_idx)
            decoded_post = lm_decode_known_good(logits_post)
        decoded_list.append(decoded_pre if score_on == "pre" else decoded_post)
        ref_list.append(ref)

    _, wer = _cer_and_wer(decoded_list, ref_list, outputType="speech", returnCI=False)
    return wer


def _device_with_most_free_memory():
    """Return cuda device index with the most free memory."""
    if not torch.cuda.is_available():
        return 0
    n = torch.cuda.device_count()
    if n == 0:
        return 0
    if hasattr(torch.cuda, "mem_get_info"):
        best = 0
        best_free = 0
        for i in range(n):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best_free = free
                best = i
        return best
    # Fallback: use cuda:1 if multiple GPUs (cuda:0 often full from other use)
    return 1 if n > 1 else 0


def main():
    parser = argparse.ArgumentParser(description="TTA sweep: val + test (no val init + val init)")
    parser.add_argument("--split", type=str, default="split1")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' (pick GPU with most free memory), 'cuda:0', 'cuda:1', ... or 'cpu'")
    parser.add_argument("--multi-gpu", action="store_true", dest="multi_gpu",
                        help="Run one seed per GPU in subprocesses (avoids OOM when LM fills one GPU).")
    parser.add_argument("--results-dir", type=str, default=os.path.join(BASE, "results", "wer_t15_new_tta"))
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=123)
    parser.add_argument("--ours-tta-lrs", type=float, nargs="+", default=[0.03])
    parser.add_argument("--gru-tta-lr", type=float, default=0.02, help="GRU TTA learning rate")
    args = parser.parse_args()

    if args.device == "auto":
        device = None  # set after LM decoder load (LM may use cuda:0 and fill it)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = ev.split_config(args.split)
    baseline_dir = Path(cfg["baseline_dir"] + "/checkpoints")
    ours_dir = Path(cfg["ours_dir"])
    val_sessions = cfg["val_sess"]
    test_sessions = cfg["test_sess"]
    last_day_idx = cfg["last_day_idx"]
    last_val_day_idx = ev.SESSIONS_26.index(val_sessions[0])
    sessions_eval = val_sessions
    sessions_test = test_sessions
    data_subdir = cfg["data_subdir"]
    gru_tta_lr = args.gru_tta_lr

    # Multi-GPU: one subprocess per seed with CUDA_VISIBLE_DEVICES (like plot_n_gram_wer_eval.py)
    if getattr(args, "multi_gpu", False) and args.num_seeds > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        n_gpus = torch.cuda.device_count()
        seeds = [args.seed0 + i for i in range(args.num_seeds)]
        argv_base = [
            sys.executable, __file__,
            "--split", args.split,
            "--results-dir", args.results_dir,
            "--num-seeds", "1",
            "--device", "cuda:0",
            "--gru-tta-lr", str(args.gru_tta_lr),
        ]
        argv_base += ["--ours-tta-lrs"] + [str(lr) for lr in args.ours_tta_lrs]
        # Use GPUs 1, 2, 3, ... (skip GPU 0 which is often full from LM or other processes)
        gpu_ids = list(range(1, n_gpus)) if n_gpus > 1 else [0]
        procs = []
        for s, seed in enumerate(seeds):
            gpu_id = gpu_ids[s % len(gpu_ids)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            argv = argv_base + ["--seed0", str(seed)]
            p = subprocess.Popen(argv, env=env, cwd=REPO)
            procs.append((s, seed, gpu_id, p))
        print(f"Multi-GPU: running {len(seeds)} seeds on GPUs {gpu_ids} (skipping GPU 0)", flush=True)
        for s, seed, gpu_id, p in procs:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Subprocess seed={seed} (GPU {gpu_id}) exited with {p.returncode}")
        print("Multi-GPU: all subprocesses finished.", flush=True)
        return

    ev._ensure_lm_decoder()
    # Pick GPU with most free memory *after* LM decoder load (LM often fills cuda:0)
    if device is None:
        if torch.cuda.is_available():
            idx = _device_with_most_free_memory()
            device = torch.device(f"cuda:{idx}")
            print(f"Using device: {device} (GPU with most free memory after LM load)", flush=True)
        else:
            device = torch.device("cpu")
    args_base = ev._load_yaml(str(baseline_dir / "args.yaml"))
    args_ours = ev._load_yaml(str(ours_dir / "args.yaml"))
    args_base.setdefault("dataset", {})
    args_ours.setdefault("dataset", {})
    args_base["dataset"]["sessions"] = ev.SESSIONS_26
    args_ours["dataset"]["sessions"] = ev.SESSIONS_26

    ckpt_ours = ev._resolve_best_checkpoint(ours_dir)
    ckpt_gru = ev._resolve_best_checkpoint(baseline_dir)
    tta_args = {
        "repeats": [64],
        "adaptation_steps": 1,
        "WN+BS": True,
        "white_noise": 1.0,
        "baseline_shift": 0.2,
        "l2_decay": 0.001,
    }

    ours_tta_lrs = args.ours_tta_lrs
    seeds = [args.seed0 + i for i in range(args.num_seeds)]
    n_val_days = len(sessions_eval)
    n_test_days = len(sessions_test)
    loaders_per_val_day = [
        ev._make_eval_loader_from_manifest(args_base, [sessions_eval[d]], split_name="val")
        for d in range(n_val_days)
    ]
    loaders_per_test_day = [
        ev._make_eval_loader_from_manifest(args_base, [sessions_test[d]], split_name="test")
        for d in range(n_test_days)
    ]

    results_test = [(d, sessions_test[d], len(loaders_per_test_day[d]), {lr: [] for lr in ours_tta_lrs}) for d in range(n_test_days)]
    results_test_valinit = [(d, sessions_test[d], len(loaders_per_test_day[d]), {lr: [] for lr in ours_tta_lrs}) for d in range(n_test_days)]
    results_test_gru = [(d, sessions_test[d], len(loaders_per_test_day[d]), []) for d in range(n_test_days)]
    results_test_valinit_gru = [(d, sessions_test[d], len(loaders_per_test_day[d]), []) for d in range(n_test_days)]

    save_dir = os.path.join(args.results_dir, data_subdir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}", flush=True)

    cpu_device = torch.device("cpu")
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for s, seed in enumerate(seeds):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # ---------- Ours only on GPU (then free) ----------
        model_ours = ev._build_gru_mdan_from_args(args_ours, cpu_device)
        ev._load_ckpt_state(model_ours, str(ckpt_ours), cpu_device, fill_missing_day_from_idx=last_val_day_idx)
        ev._set_eval_dropouts_to_zero(model_ours)
        if device.type == "cuda":
            model_ours = model_ours.to(device)

        # Val (cumulative over val days; store per-day WER)
        val_wer_per_day_valinit_pass_ours = {lr: [] for lr in ours_tta_lrs}
        ev._load_ckpt_state(model_ours, str(ckpt_ours), device, fill_missing_day_from_idx=last_val_day_idx)
        ev._set_eval_dropouts_to_zero(model_ours)
        for d in range(n_val_days):
            loader_d = loaders_per_val_day[d]
            for lr in ours_tta_lrs:
                wer = run_gru_tta_day_only(
                    model=model_ours, forced_day=last_day_idx, loader=loader_d,
                    tta_args=tta_args, lr=lr, device=device, debug_first_n=0,
                )
                val_wer_per_day_valinit_pass_ours[lr].append(wer)
        ckpt_valinit_path = os.path.join(save_dir, f"model_post_val_tta_seed{seed}.pth")
        torch.save(model_ours.state_dict(), ckpt_valinit_path)
        print(f"  Saved ours val-init ckpt: {ckpt_valinit_path}", flush=True)
        # Test (no val init, cumulative)
        ev._load_ckpt_state(model_ours, str(ckpt_ours), device, fill_missing_day_from_idx=last_val_day_idx)
        ev._set_eval_dropouts_to_zero(model_ours)
        for d in range(n_test_days):
            loader_d = loaders_per_test_day[d]
            for lr in ours_tta_lrs:
                wer = run_gru_tta_day_only(
                    model=model_ours, forced_day=last_day_idx, loader=loader_d,
                    tta_args=tta_args, lr=lr, device=device, debug_first_n=0,
                )
                results_test[d][3][lr].append(wer)
        # Test (val init, cumulative)
        model_ours.load_state_dict(torch.load(ckpt_valinit_path, map_location=device))
        ev._set_eval_dropouts_to_zero(model_ours)
        for d in range(n_test_days):
            loader_d = loaders_per_test_day[d]
            for lr in ours_tta_lrs:
                wer = run_gru_tta_day_only(
                    model=model_ours, forced_day=last_day_idx, loader=loader_d,
                    tta_args=tta_args, lr=lr, device=device, debug_first_n=0,
                )
                results_test_valinit[d][3][lr].append(wer)
        del model_ours
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---------- GRU only on GPU (then free) ----------
        model_gru = ev._build_gru_from_args(args_base, cpu_device)
        ev._load_ckpt_state(model_gru, str(ckpt_gru), cpu_device, fill_missing_day_from_idx=last_day_idx)
        ev._set_eval_dropouts_to_zero(model_gru)
        if device.type == "cuda":
            model_gru = model_gru.to(device)

        # Val (cumulative; store per-day WER)
        val_wer_per_day_valinit_pass_gru = []
        ev._load_ckpt_state(model_gru, str(ckpt_gru), device, fill_missing_day_from_idx=last_day_idx)
        ev._set_eval_dropouts_to_zero(model_gru)
        for d in range(n_val_days):
            loader_d = loaders_per_val_day[d]
            wer = run_gru_tta_day_only(
                model=model_gru, forced_day=last_day_idx, loader=loader_d,
                tta_args=tta_args, lr=gru_tta_lr, device=device, debug_first_n=0,
            )
            val_wer_per_day_valinit_pass_gru.append(wer)
        ckpt_valinit_gru_path = os.path.join(save_dir, f"model_post_val_tta_gru_seed{seed}.pth")
        torch.save(model_gru.state_dict(), ckpt_valinit_gru_path)
        print(f"  Saved GRU val-init ckpt: {ckpt_valinit_gru_path}", flush=True)
        # Test (no val init, cumulative)
        ev._load_ckpt_state(model_gru, str(ckpt_gru), device, fill_missing_day_from_idx=last_day_idx)
        ev._set_eval_dropouts_to_zero(model_gru)
        for d in range(n_test_days):
            loader_d = loaders_per_test_day[d]
            wer = run_gru_tta_day_only(
                model=model_gru, forced_day=last_day_idx, loader=loader_d,
                tta_args=tta_args, lr=gru_tta_lr, device=device, debug_first_n=0,
            )
            results_test_gru[d][3].append(wer)
        # Test (val init, cumulative)
        model_gru.load_state_dict(torch.load(ckpt_valinit_gru_path, map_location=device))
        ev._set_eval_dropouts_to_zero(model_gru)
        for d in range(n_test_days):
            loader_d = loaders_per_test_day[d]
            wer = run_gru_tta_day_only(
                model=model_gru, forced_day=last_day_idx, loader=loader_d,
                tta_args=tta_args, lr=gru_tta_lr, device=device, debug_first_n=0,
            )
            results_test_valinit_gru[d][3].append(wer)
        del model_gru
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for lr in ours_tta_lrs:
            print(f"  Val-init pass ours lr={lr} WER per day: " + ", ".join(f"{w:.2%}" for w in val_wer_per_day_valinit_pass_ours[lr]), flush=True)
        print(f"  Val-init pass GRU lr={gru_tta_lr} WER per day: " + ", ".join(f"{w:.2%}" for w in val_wer_per_day_valinit_pass_gru), flush=True)

        wer_save = {
            "lrs": ours_tta_lrs,
            "gru_tta_lr": gru_tta_lr,
            "seed": seed,
            "val_wer_per_day_per_lr": {str(lr): val_wer_per_day_valinit_pass_ours[lr] for lr in ours_tta_lrs},
            "val_wer_per_day_gru": val_wer_per_day_valinit_pass_gru,
            "val_session_names": list(sessions_eval),
            "val_n_trials": [len(loaders_per_val_day[d]) for d in range(n_val_days)],
            "test_wer_per_day_per_lr": {str(lr): [results_test[d][3][lr][s] for d in range(n_test_days)] for lr in ours_tta_lrs},
            "test_wer_per_day_gru": [results_test_gru[d][3][s] for d in range(n_test_days)],
            "test_wer_per_day_per_lr_valinit": {str(lr): [results_test_valinit[d][3][lr][s] for d in range(n_test_days)] for lr in ours_tta_lrs},
            "test_wer_per_day_gru_valinit": [results_test_valinit_gru[d][3][s] for d in range(n_test_days)],
            "test_session_names": [r[1] for r in results_test],
            "test_n_trials": [r[2] for r in results_test],
        }
        save_path = os.path.join(save_dir, f"wer_tta_seed{seed}.json")
        with open(save_path, "w") as f:
            json.dump(wer_save, f, indent=2)
        print(f"Saved seed {seed} ({s+1}/{len(seeds)}) to {save_path}", flush=True)

    print("--- Test no val-init (ours) ---", flush=True)
    for d in range(n_test_days):
        session_name, n_d, wers_for_day = results_test[d][1], results_test[d][2], results_test[d][3]
        for lr in ours_tta_lrs:
            wers = wers_for_day[lr]
            print(f"Test day {d} ({session_name}, n={n_d}): ours lr={lr} WER: " + ", ".join(f"{w:.2%}" for w in wers)
                  + f" | mean±std: {np.mean(wers):.2%} ± {np.std(wers):.2%}", flush=True)
    print("--- Test no val-init (GRU) ---", flush=True)
    for d in range(n_test_days):
        session_name, n_d, wers_list = results_test_gru[d][1], results_test_gru[d][2], results_test_gru[d][3]
        print(f"Test day {d} ({session_name}, n={n_d}): GRU lr={gru_tta_lr} WER: " + ", ".join(f"{w:.2%}" for w in wers_list)
              + f" | mean±std: {np.mean(wers_list):.2%} ± {np.std(wers_list):.2%}", flush=True)
    print("--- Test val-init (ours) ---", flush=True)
    for d in range(n_test_days):
        session_name, n_d, wers_for_day = results_test_valinit[d][1], results_test_valinit[d][2], results_test_valinit[d][3]
        for lr in ours_tta_lrs:
            wers = wers_for_day[lr]
            print(f"Test valinit day {d} ({session_name}, n={n_d}): ours lr={lr} WER: " + ", ".join(f"{w:.2%}" for w in wers)
                  + f" | mean±std: {np.mean(wers):.2%} ± {np.std(wers):.2%}", flush=True)
    print("--- Test val-init (GRU) ---", flush=True)
    for d in range(n_test_days):
        session_name, n_d, wers_list = results_test_valinit_gru[d][1], results_test_valinit_gru[d][2], results_test_valinit_gru[d][3]
        print(f"Test valinit day {d} ({session_name}, n={n_d}): GRU lr={gru_tta_lr} WER: " + ", ".join(f"{w:.2%}" for w in wers_list)
              + f" | mean±std: {np.mean(wers_list):.2%} ± {np.std(wers_list):.2%}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
