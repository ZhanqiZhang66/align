#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch

# -----------------------------
# Path / env setup
# -----------------------------
BASE = "/victoriapvc"
REPO = os.path.join(BASE, "repos", "brain2text-t15")
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(BASE, "pip_packages"))

os.environ.setdefault("LM_DECODER_DIR", "/victoriapvc/data/willett/lm/languageModel")
os.environ.setdefault("BRAIN2TEXT_LM_DIR", "/victoriapvc/data/willett/lm/languageModel")

from tta_utils import clean_transcription, get_phonemes
from lm_utils import _cer_and_wer
from model_training.data_augmentations import gauss_smooth
import run_n_gram_wer_eval as ev


# -----------------------------
# Small utils (copy of yours)
# -----------------------------
def transcription_tensor_to_str(t):
    a = t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if (a == 0).any():
        a = a[: int(np.where(a == 0)[0][0])]
    return bytes(a.tolist()).decode("utf-8", errors="ignore")


def freeze_day_affine_only(model):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("day_weights.") or name.startswith("day_biases."):
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No day_weights/day_biases params found. Check parameter names.")
    return trainable


def lm_decode_one(logits_1tC, ev_mod):
    T = int(logits_1tC.shape[1])
    decoded, _, _ = ev_mod.get_lm_outputs(logits_1tC, n_frames=T)
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    return clean_transcription(decoded)


def pick_freest_gpu():
    """Return torch.device for GPU with most free memory."""
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_i = i
    return torch.device(f"cuda:{best_i}")


def run_gru_tta_streaming(
    *,
    model,
    forced_day: int,
    loader,
    tta_args: dict,
    device,
    ev_mod,
    optimizer,
    trainable_params,
    debug_first_n: int = 0,
    score_on: str = "pre",  # "pre" or "post"
):
    assert score_on in {"pre", "post"}

    n_augs = int(tta_args["repeats"][0])
    adapt_steps = int(tta_args["adaptation_steps"])
    use_aug = bool(tta_args.get("WN+BS", False))

    white_noise_std = float(tta_args.get("white_noise_std", tta_args.get("white_noise", 1.0)))
    constant_offset_std = float(tta_args.get("constant_offset_std", tta_args.get("baseline_shift", 0.2)))
    static_gain_std = float(tta_args.get("static_gain_std", 0.0))
    random_walk_std = float(tta_args.get("random_walk_std", 0.0))
    random_walk_axis = int(tta_args.get("random_walk_axis", -1))
    random_cut = int(tta_args.get("random_cut", 0))
    smooth_data = bool(tta_args.get("smooth_data", False))
    smooth_kernel_std = float(tta_args.get("smooth_kernel_std", 2.0))
    smooth_kernel_size = int(tta_args.get("smooth_kernel_size", 100))

    forced_day_t = torch.tensor([int(forced_day)], device=device, dtype=torch.long)

    decoded_list, ref_list = [], []

    for batch_idx, batch in enumerate(loader):
        features = batch["input_features"].to(device)
        B = int(features.shape[0])
        assert B == 1, f"Expected B=1 streaming loader, got B={B}"

        day_idx = forced_day_t

        ref_raw = transcription_tensor_to_str(batch["transcriptions"][0])
        ref = clean_transcription(ref_raw)

        # PRE decode
        model.eval()
        with torch.no_grad():
            logits_pre = model(features, day_idx)
            decoded_pre = lm_decode_one(logits_pre, ev_mod)

        if batch_idx < debug_first_n:
            print(f"\n[tta] trial {batch_idx}")
            print("  REF :", ref)
            print("  PRE :", decoded_pre)

        # pseudo labels
        y_pseudo, y_len_pseudo = get_phonemes(decoded_pre)
        if not torch.is_tensor(y_pseudo):
            y_pseudo = torch.tensor(y_pseudo, dtype=torch.long)
        y_pseudo = y_pseudo.to(device)
        y_len_pseudo = torch.tensor([int(y_len_pseudo)], device=device, dtype=torch.int32)

        # TTA update
        if adapt_steps > 0:
            if n_augs > 1:
                X = features.repeat(n_augs, 1, 1)
                day_rep = day_idx.repeat(n_augs)
                y_rep = y_pseudo.unsqueeze(0).repeat(n_augs, 1)
                ylen_rep = y_len_pseudo.repeat(n_augs)
            else:
                X = features
                day_rep = day_idx
                y_rep = y_pseudo.unsqueeze(0)
                ylen_rep = y_len_pseudo

            if use_aug and n_augs > 1:
                BB, Tt, Cc = X.shape
                if static_gain_std > 0:
                    warp_mat = torch.eye(Cc, device=device).unsqueeze(0).repeat(BB, 1, 1)
                    warp_mat = warp_mat + torch.randn_like(warp_mat) * static_gain_std
                    X = torch.matmul(X, warp_mat)
                if white_noise_std > 0:
                    X = X + torch.randn_like(X) * white_noise_std
                if constant_offset_std > 0:
                    X = X + torch.randn((BB, 1, Cc), device=device) * constant_offset_std
                if random_walk_std > 0:
                    X = X + torch.cumsum(torch.randn_like(X) * random_walk_std, dim=random_walk_axis)
                if random_cut > 0:
                    cut = np.random.randint(0, random_cut)
                    X = X[:, cut:, :]
                if smooth_data:
                    X = gauss_smooth(
                        X,
                        device=device,
                        smooth_kernel_std=smooth_kernel_std,
                        smooth_kernel_size=smooth_kernel_size,
                    )

            model.train()
            for _ in range(adapt_steps):
                logits = model(X, day_rep)
                Tlog = int(logits.shape[1])
                logit_lens = torch.full((logits.shape[0],), Tlog, device=device, dtype=torch.int32)

                loss = ev_mod._ctc_loss_exact(logits, y_rep, logit_lens, ylen_rep)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()

        # POST decode (optional)
        model.eval()
        with torch.no_grad():
            logits_post = model(features, day_idx)
            decoded_post = lm_decode_one(logits_post, ev_mod)

        if batch_idx < debug_first_n:
            print("  POST:", decoded_post)

        decoded_list.append(decoded_pre if score_on == "pre" else decoded_post)
        ref_list.append(ref)

    _, wer = _cer_and_wer(decoded_list, ref_list, outputType="speech", returnCI=False)
    return wer


# -----------------------------
# Main runner
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="split1")
    ap.add_argument("--output-dir", type=str, required=True)

    ap.add_argument("--last-val-idx", type=int, required=True, help="forced_day index used for GRU forward (e.g. 4)")
    ap.add_argument("--val-day-idx", type=int, required=True, help="which val_day_idx was used when saving ckpt filenames")

    ap.add_argument("--gru-lr", type=float, required=True)
    ap.add_argument("--ours-lr", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)

    ap.add_argument("--ckpt-pattern-gru", type=str, required=True)
    ap.add_argument("--ckpt-pattern-ours", type=str, required=True)

    ap.add_argument("--score-on", type=str, default="pre", choices=["pre", "post"])
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help='CUDA device string (e.g. "cuda:0"); use "auto" to pick the freest GPU',
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        if args.device.lower() in {"auto", "cuda", "cuda:auto"}:
            device = pick_freest_gpu()
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config + loaders
    cfg = ev.split_config(args.split)

    baseline_dir = Path(cfg["baseline_dir"] + "/checkpoints")
    ours_dir = Path(cfg["ours_dir"])

    # We build loaders using args_ours like you did (sessions list = SESSIONS_26)
    args_base = ev._load_yaml(str(baseline_dir / "args.yaml"))
    args_ours = ev._load_yaml(str(ours_dir / "args.yaml"))
    args_base.setdefault("dataset", {})
    args_ours.setdefault("dataset", {})
    args_base["dataset"]["sessions"] = ev.SESSIONS_26
    args_ours["dataset"]["sessions"] = ev.SESSIONS_26

    test_sessions = cfg["test_sess"]
    sessions_test = test_sessions[:4]
    n_test_days = len(sessions_test)
    print(f"Test sessions ({n_test_days}): {sessions_test}")

    loaders_per_test_day = [
        ev._make_eval_loader_from_manifest(args_ours, [sessions_test[d]], split_name="test")
        for d in range(n_test_days)
    ]

    # TTA args (keep exactly as your snippet; edit here if you want)
    tta_args = {
        "repeats": [64],
        "adaptation_steps": 1,
        "WN+BS": True,
        "white_noise": 1.0,
        "baseline_shift": 0.2,
        "l2_decay": 0.0,
        "random_walk_std": 0.0,
        "random_walk_axis": -1,
        "static_gain_std": 0.0,
        "random_cut": 3,
        "smooth_data": True,
        "smooth_kernel_std": 2,
        "smooth_kernel_size": 100,
    }
    wd = float(tta_args.get("l2_decay", 0.0))

    for seed in args.seeds:
        print(f"\n==============================")
        print(f"Seed {seed}")
        print(f"==============================")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ---------
        # GRU baseline: build + load YOUR saved ckpt
        # ---------
        model_gru = ev._build_gru_from_args(args_base, device)
        trainable_gru = freeze_day_affine_only(model_gru)

        ckpt_name_gru = args.ckpt_pattern_gru.format(
            val_day_idx=args.val_day_idx,
            lr=args.gru_lr,
            seed=seed,
        )
        ckpt_path_gru = out_dir / ckpt_name_gru
        if not ckpt_path_gru.exists():
            raise FileNotFoundError(f"Missing GRU ckpt: {ckpt_path_gru}")

        ev._load_ckpt_state(
            model_gru,
            str(ckpt_path_gru),
            device,
            fill_missing_day_from_idx=args.last_val_idx,
        )

        opt_gru = torch.optim.AdamW(
            trainable_gru,
            lr=args.gru_lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=0.1,
        )

        for test_day_idx in range(n_test_days):
            wer = run_gru_tta_streaming(
                model=model_gru,
                forced_day=args.last_val_idx,
                loader=loaders_per_test_day[test_day_idx],
                tta_args=tta_args,
                device=device,
                ev_mod=ev,
                optimizer=opt_gru,
                trainable_params=trainable_gru,
                debug_first_n=0,
                score_on=args.score_on,
            )
            print(f"[GRU] test day {test_day_idx} WER: {wer:.2%}")

            out_json = out_dir / f"gru_from_savedckpt_test_day{test_day_idx}_lr{args.gru_lr}_seed{seed}.json"
            with open(out_json, "w") as f:
                json.dump({"test_day_index": test_day_idx, "wer": float(wer)}, f, indent=2)

        # ---------
        # Ours / ALIGN: build + load YOUR saved ckpt
        # NOTE: you used _build_gru_from_args(args_base, ...) in your snippet,
        # but if ALIGN model differs, swap builder/args here accordingly.
        # ---------
        model_ours = ev._build_gru_from_args(args_base, device)
        trainable_ours = freeze_day_affine_only(model_ours)

        ckpt_name_ours = args.ckpt_pattern_ours.format(
            val_day_idx=args.val_day_idx,
            lr=args.ours_lr,
            seed=seed,
        )
        ckpt_path_ours = out_dir / ckpt_name_ours
        if not ckpt_path_ours.exists():
            raise FileNotFoundError(f"Missing OURS ckpt: {ckpt_path_ours}")

        ev._load_ckpt_state(
            model_ours,
            str(ckpt_path_ours),
            device,
            fill_missing_day_from_idx=args.last_val_idx,
        )

        opt_ours = torch.optim.AdamW(
            trainable_ours,
            lr=args.ours_lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=0.1,
        )

        for test_day_idx in range(n_test_days):
            wer = run_gru_tta_streaming(
                model=model_ours,
                forced_day=args.last_val_idx,
                loader=loaders_per_test_day[test_day_idx],
                tta_args=tta_args,
                device=device,
                ev_mod=ev,
                optimizer=opt_ours,
                trainable_params=trainable_ours,
                debug_first_n=0,
                score_on=args.score_on,
            )
            print(f"[OURS/ALIGN] test day {test_day_idx} WER: {wer:.2%}")

            out_json = out_dir / f"align_from_savedckpt_test_day{test_day_idx}_lr{args.ours_lr}_seed{seed}.json"
            with open(out_json, "w") as f:
                json.dump({"test_day_index": test_day_idx, "wer": float(wer)}, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()


"""
#!/usr/bin/env bash
set -euo pipefail

# If you want exactly 5 hours:
sleep $((5 * 60 * 60))

cd /victoriapvc/repos/brain2text-t15
export LM_DECODER_DIR=/victoriapvc/data/willett/lm/languageModel
export BRAIN2TEXT_LM_DIR=/victoriapvc/data/willett/lm/languageModel
export PYTHONPATH=/victoriapvc/repos/brain2text-t15:/victoriapvc/pip_packages

python run_test_from_saved_ckpt.py \
  --split split1 \
  --output-dir /victoriapvc/results/t15-final/split1 \
  --last-val-idx 4 \
  --ckpt-pattern-gru "gru_val_day{val_day_idx}_lr{lr}_seed{seed}_ckpt.pth" \
  --ckpt-pattern-ours "ours_val_day{val_day_idx}_lr{lr}_seed{seed}_ckpt.pth" \
  --val-day-idx 0 \
  --gru-lr 0.005 \
  --ours-lr 0.0005 \
  --seeds 1 2 3 4 5



tmux new -s tta_val_init_test_split1 '
cd /victoriapvc/repos/brain2text-t15 && \
export LM_DECODER_DIR=/victoriapvc/data/willett/lm/languageModel && \
export BRAIN2TEXT_LM_DIR=/victoriapvc/data/willett/lm/languageModel && \
export PYTHONPATH=/victoriapvc/repos/brain2text-t15:/victoriapvc/pip_packages && \
python run_val_init.py \
  --split split1 \
  --output-dir /victoriapvc/results/t15-final/split1 \
  --last-val-idx 4 \
  --ckpt-pattern-gru "gru_val_day{val_day_idx}_lr{lr}_seed{seed}_ckpt.pth" \
  --ckpt-pattern-ours "ours_val_day{val_day_idx}_lr{lr}_seed{seed}_ckpt.pth" \
  --val-day-idx 12 \
  --gru-lr 0.005 \
  --ours-lr 0.0005 \
  --seeds 1 2 3 4 5 ; \
echo "EXIT CODE=$?"; read -p "Press Enter to close..."'
"""