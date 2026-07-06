#!/usr/bin/env python3
"""
# No TTA: 1 run per model (val + test). TTA: --n-runs 5 seeds per LR. LRs: edit --ours-tta-lrs if needed.
# Example (split1, 4 LRs, recompute everything):
tmux new -s wer_t15_new 'cd /victoriapvc && \
export LM_DECODER_DIR=/victoriapvc/data/willett/lm/languageModel && \
export BRAIN2TEXT_LM_DIR=/victoriapvc/data/willett/lm/languageModel && \
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
PYTHONPATH=/victoriapvc/repos/brain2text-t15:/victoriapvc/pip_packages \
python repos/brain2text-t15/run_n_gram_wer_eval.py \
  --split split1 --device cuda:0 --no-cache --n-runs 5 --ours-tta-lrs 2e-2 1e-2 1e-1 2e-3 \
  --results-dir /victoriapvc/results/wer_t15_new; \
echo "EXIT CODE=$?"; read -p "Press Enter to close"'

tmux new -s wer 'cd /victoriapvc/repos/brain2text-t15 && \
PYTHONPATH=/victoriapvc/repos/brain2text-t15:/victoriapvc/pip_packages \
python run_n_gram_wer_eval.py --no-cache --wer-debug \
--split split1 --ours-tta-lrs 2e-2 \
--replot-only
--results-dir /victoriapvc/results/wer_t15_new; \
echo "EXIT CODE=$?"; read -p "Press Enter to close"'
T15 n-gram WER eval with T12-style logic/format — MANIFEST-ONLY (matches your trainer).

Key change you requested:
ALWAYS use manifest (no dataset_dir fallback).
Eval loader matches trainer exactly:
   - val_dataset uses BrainToTextDataset(..., split='test', days_per_batch=None, n_batches=None, shuffle=False, num_workers=0)
trial_indicies come from manifest and are remapped rel->abs exactly like training snippet.

Directory layout (per split):
  baseline_dir/
	checkpoints/
		args.yaml
		best_checkpoint
		val_metrics.pkl
		train_val_trials.json
		training_log
  ours_dir/
	args.yaml
	best_checkpoint
	val_metrics.pkl
	...

USAGE:
python run_n_gram_wer_eval.py \
  --split split1 \
  --device cuda:0 \
  --n-runs 5 \
  --ours-tta-lrs 3e-4 5e-4 1e-3 3e-3 \
  --results-dir /victoriapvc/results/wer_t15_new
"""

import argparse
import copy
import datetime
import hashlib
import os
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
# -----------------------------
# Sessions (must match model indexing order)
# -----------------------------
SESSIONS_26 = [
	"t15.2023.08.11", "t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20",
	"t15.2023.08.25", "t15.2023.08.27", "t15.2023.09.01", "t15.2023.09.03",
	"t15.2023.09.24", "t15.2023.09.29", "t15.2023.10.01", "t15.2023.10.06",
	"t15.2023.10.08", "t15.2023.10.13", "t15.2023.10.15", "t15.2023.10.20",
	"t15.2023.10.22", "t15.2023.11.03", "t15.2023.11.04", "t15.2023.11.17",
	"t15.2023.11.19", "t15.2023.11.26", "t15.2023.12.03", "t15.2023.12.08",
	"t15.2023.12.10", "t15.2023.12.17",
]

# -----------------------------
# T15 LM hyperparams (fixed)
# -----------------------------
LM_BEAM = 17
ACOUSTIC_SCALE = 0.5
BLANK_PENALTY = float(np.log(9.0))
N_BEST = 1
RETURN_N_BEST = False
RESCORE = True

LOAD_LM = True
ngramDecoder = None
LM_CAN_RESCORE = None

BASE_DIR = "/victoriapvc"
LM_DECODER_SO_DIR = "/victoriapvc/data/willett/lm/languageModel"
DEFAULT_LARGE_LM_DIR = "/victoriapvc/data/willett/lm/languageModel"

# Keep decoder .so resolution stable across lm_utils variants.
os.environ.setdefault("LM_DECODER_DIR", LM_DECODER_SO_DIR)


# ---------------------------------------------------------------------------
# Split config (your requested style)
# ---------------------------------------------------------------------------
def split_config(split: str):
	if split == "split1":
		return dict(
			data_subdir="5train_13val_8test",
			last_day_idx=4,
			train_sess=SESSIONS_26[:5],
			val_sess=SESSIONS_26[5:5 + 13],
			test_sess=SESSIONS_26[-8:],
			baseline_dir="/victoriapvc/data/outputs/baseline_split1",
			ours_dir="/victoriapvc/data/outputs/20260218_s011_rep2_lam0.7_lrm0.8_wup5_hid256_dcp0_dd0_alpalt_alf24",
		)
	if split == "split2":
		return dict(
			data_subdir="10train_8val_8test",
			last_day_idx=9,
			train_sess=SESSIONS_26[:10],
			val_sess=SESSIONS_26[10:10 + 8],
			test_sess=SESSIONS_26[-8:],
			baseline_dir="/victoriapvc/data/outputs/baseline_split2",
			ours_dir="/victoriapvc/data/outputs/ours_split2",
		)
	if split == "split3":
		return dict(
			data_subdir="5train_5val_8test",
			last_day_idx=4,
			train_sess=SESSIONS_26[:5],
			val_sess=SESSIONS_26[5:5 + 5],
			test_sess=SESSIONS_26[10: 10 + 8],
			baseline_dir="/victoriapvc/data/outputs/baseline_split1",
			ours_dir="/victoriapvc/data/brain2text_25/outputs",
		)
	if split == "split5":
		return dict(
			data_subdir="10train_5val_5test",
			last_day_idx=9,
			train_sess=SESSIONS_26[:10],
			val_sess=SESSIONS_26[10:10 + 5],
			test_sess=SESSIONS_26[15:15 + 5],
			baseline_dir="/victoriapvc/data/outputs/checkpoints_seed0",
			ours_dir="/victoriapvc/data/brain2text_25/outputs", # TODO: change to our own model
		)
	raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _parse_t15_date(sess: str) -> datetime.date:
	_, y, m, d = sess.split(".")
	return datetime.date(int(y), int(m), int(d))


def _lighten_rgb(color_rgb, amount=0.55):
	r, g, b = color_rgb
	return (
		1 - (1 - r) * (1 - amount),
		1 - (1 - g) * (1 - amount),
		1 - (1 - b) * (1 - amount),
	)


def _md5_of_obj(obj) -> str:
	key_bytes = pickle.dumps(obj, protocol=0)
	return hashlib.md5(key_bytes).hexdigest()


def _run_cache_path(cache_dir: Path, split_name: str, tta_mode: str, seed: int, ours_lr: Optional[float], extra_key: dict) -> Path:
	h = _md5_of_obj((split_name, tta_mode, seed, ours_lr, extra_key))[:14]
	lr_tag = "nolr" if ours_lr is None else f"lr{ours_lr:g}"
	return cache_dir / f"{split_name}_{tta_mode}_{lr_tag}_seed{seed}_{h}.pkl"


def _find_any_val_cache(cache_dir: Path, split_name: str, tta_mode: str, seed: int, ours_lr: Optional[float]) -> Optional[Path]:
	"""For val only: find any cache file with same split, tta_mode, seed, lr (ignore hash)."""
	if split_name != "val":
		return None
	lr_tag = "nolr" if ours_lr is None else f"lr{ours_lr:g}"
	pattern = f"{split_name}_{tta_mode}_{lr_tag}_seed{seed}_*.pkl"
	matches = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
	return matches[0] if matches else None


def _load_cached_run(path: Path):
	if not path.exists():
		return None
	try:
		with open(path, "rb") as f:
			return pickle.load(f)
	except Exception as e:
		print(f"[cache] load failed: {path} ({e})")
		return None


def _save_cached_run(path: Path, out: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	try:
		with open(path, "wb") as f:
			pickle.dump(out, f)
		print(f"[cache] saved: {path}")
	except Exception as e:
		print(f"[cache] save failed: {path} ({e})")


def _get_from_nested(d: dict, keys: List[str], default=None):
	cur = d
	for k in keys:
		if not isinstance(cur, dict) or k not in cur:
			return default
		cur = cur[k]
	return cur


def _transcription_tensor_to_str(t) -> str:
	"""Convert batch['transcriptions'][i] tensor (bytes/chars) to UTF-8 string."""
	a = t.detach().cpu().numpy()
	if a.dtype != np.uint8:
		a = a.astype(np.uint8)
	if (a == 0).any():
		a = a[: int(np.where(a == 0)[0][0])]
	return bytes(a.tolist()).decode("utf-8", errors="ignore")


def _resolve_lm_dir() -> str:
	# Keep behavior aligned with the stable T12 script:
	# always use Willett LM directory unless user explicitly overrides.
	lm_dir = os.environ.get("BRAIN2TEXT_LM_DIR") or DEFAULT_LARGE_LM_DIR
	if os.path.exists(os.path.join(lm_dir, "TLG.fst")) and os.path.exists(os.path.join(lm_dir, "words.txt")):
		return lm_dir
	raise FileNotFoundError(
		"Could not find LM directory with TLG.fst/words.txt. "
		"Set BRAIN2TEXT_LM_DIR to a valid language model directory."
	)


# ---------------------------------------------------------------------------
# Checkpoint discovery (best_checkpoint layout)
# ---------------------------------------------------------------------------
def _read_text_file(p: Path) -> str:
	try:
		return p.read_text(encoding="utf-8").strip()
	except Exception:
		return ""


def _resolve_best_checkpoint(checkpoints_dir: Path) -> Path:
	best = checkpoints_dir / "best_checkpoint"
	if not best.exists():
		raise FileNotFoundError(f"Missing {best}")

	try:
		if best.is_symlink():
			target = best.resolve()
			if target.exists():
				return target
	except Exception:
		pass

	if best.suffix in {".pt", ".pth", ".ckpt"}:
		return best

	txt = _read_text_file(best)
	if txt:
		cand = Path(txt)
		if not cand.is_absolute():
			cand = checkpoints_dir / cand
		if cand.exists():
			return cand
		for ext in (".pt", ".pth", ".ckpt"):
			cand2 = checkpoints_dir / (txt + ext)
			if cand2.exists():
				return cand2

	# Some runs save the checkpoint payload directly as "best_checkpoint"
	# (no extension, not a symlink, not a text pointer).
	if best.is_file() and best.stat().st_size > 0:
		try:
			_ = torch.load(str(best), map_location="cpu", weights_only=False)
			return best
		except Exception:
			pass

	ckpts = []
	for ext in ("*.pt", "*.pth", "*.ckpt"):
		ckpts.extend(checkpoints_dir.glob(ext))
	if ckpts:
		ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
		return ckpts[0]

	raise FileNotFoundError(f"Could not resolve a checkpoint in {checkpoints_dir}")


def _load_yaml(path: str) -> dict:
	from omegaconf import OmegaConf
	cfg = OmegaConf.load(path)
	return OmegaConf.to_container(cfg, resolve=True)


def _load_ckpt_state(
	model: torch.nn.Module,
	ckpt_path: str,
	device: torch.device,
	fill_missing_day_from_idx: Optional[int] = None,
):
	state = torch.load(ckpt_path, map_location=device, weights_only=False)

	if isinstance(state, dict) and "model_state_dict" in state:
		model_state = state["model_state_dict"]
	elif isinstance(state, dict) and "state_dict" in state:
		model_state = state["state_dict"]
	else:
		model_state = state

	model_state = dict(model_state)
	expected_state = model.state_dict()

	# GRU key compatibility shim:
	# - older checkpoints: gru.weight_ih_l0, gru.weight_ih_l1, ...
	# - newer checkpoints: gru_layers.0.weight_ih_l0, gru_layers.1.weight_ih_l0, ...
	exp_has_legacy_gru = any(k.startswith("gru.weight_ih_l") for k in expected_state.keys())
	exp_has_layered_gru = any(k.startswith("gru_layers.") for k in expected_state.keys())
	st_has_legacy_gru = any(k.startswith("gru.weight_ih_l") for k in model_state.keys())
	st_has_layered_gru = any(k.startswith("gru_layers.") for k in model_state.keys())

	if exp_has_legacy_gru and st_has_layered_gru and not st_has_legacy_gru:
		converted = {}
		for k, v in model_state.items():
			if k.startswith("gru_layers."):
				parts = k.split(".")
				if len(parts) >= 3 and parts[1].isdigit():
					layer_idx = int(parts[1])
					param_name = parts[2]
					if param_name.endswith("_l0"):
						base = param_name[:-3]
						converted[f"gru.{base}_l{layer_idx}"] = v
						continue
			converted[k] = v
		model_state = converted
	elif exp_has_layered_gru and st_has_legacy_gru and not st_has_layered_gru:
		converted = {}
		for k, v in model_state.items():
			if k.startswith("gru."):
				param = k.split(".", 1)[1]
				if "_l" in param:
					base, layer = param.rsplit("_l", 1)
					if layer.isdigit():
						converted[f"gru_layers.{int(layer)}.{base}_l0"] = v
						continue
			converted[k] = v
		model_state = converted

	if fill_missing_day_from_idx is not None:
		fallback_idx = int(fill_missing_day_from_idx)
		filled = []
		for prefix in ("day_weights", "day_biases"):
			src_key = f"{prefix}.{fallback_idx}"
			if src_key not in model_state:
				# Fallback to the latest available day key in checkpoint
				avail = []
				for k in model_state.keys():
					if k.startswith(f"{prefix}."):
						try:
							avail.append(int(k.split(".")[1]))
						except Exception:
							pass
				if avail:
					src_key = f"{prefix}.{max(avail)}"

			if src_key not in model_state:
				continue

			for k in expected_state.keys():
				if k.startswith(f"{prefix}.") and k not in model_state:
					model_state[k] = model_state[src_key].clone()
					filled.append((k, src_key))

		if filled:
			pass  # filled missing day-layer keys from fallback (no print)

	model.load_state_dict(model_state, strict=True)


# ---------------------------------------------------------------------------
# Manifest loading (ALWAYS)
# ---------------------------------------------------------------------------
def _load_manifest_trials(args_yaml: dict) -> Dict[str, Dict[int, object]]:
	"""
	Returns:
	  {
		"train": {abs_day_idx: trial_data, ...},
		"val": {abs_day_idx: trial_data, ...},
		"test": {abs_day_idx: trial_data, ...},
		"competition": {abs_day_idx: trial_data, ...},
		"training_days": [...],
		"target_days": [...],
	  }
	"""
	manifest_path = _get_from_nested(args_yaml, ["manifest_path"], None)
	if manifest_path is None or str(manifest_path).strip() in ("", "None"):
		raise ValueError("manifest_path is missing/None in args.yaml (but you said you always use manifest).")
	manifest_path = str(manifest_path)
	if not os.path.exists(manifest_path):
		# Prefer local formatted-data manifests when checkpoint args point to remote cluster paths.
		mname = Path(manifest_path).name
		subdir = Path(manifest_path).parent.name
		candidates = [
			f"/victoriapvc/data/brain2text_t15_formatted/{subdir}/{mname}",
			f"/victoriapvc/data/brain2text_t15_formatted/5train_13val_8test/{mname}",
			f"/victoriapvc/data/brain2text_t15_formatted/10train_8val_8test/{mname}",
		]
		found = next((p for p in candidates if os.path.exists(p)), None)
		if found is not None:
			manifest_path = found
		else:
			raise FileNotFoundError(f"manifest_path not found: {manifest_path}")

	with open(manifest_path, "rb") as f:
		manifest = pickle.load(f)

	train_trials_raw = manifest.get("train_trial_indicies", {})
	val_trials_raw = manifest.get("val_trial_indicies", {})
	test_trials_raw = manifest.get("test_trial_indicies", {})
	competition_trials_raw = manifest.get("competition_trial_indicies", {})

	training_days = manifest.get("training_days", [])
	target_days = manifest.get("target_days", [])
	test_days = manifest.get("test_days", [])

	all_sessions = args_yaml["dataset"]["sessions"]

	def remap(trials_raw: dict, day_list: List[str], fallback_offset: int):
		out = {}
		for rel_idx, trial_data in trials_raw.items():
			try:
				rel_idx_int = int(rel_idx)
			except Exception:
				continue
			if rel_idx_int < len(day_list):
				sess = day_list[rel_idx_int]
				if sess in all_sessions:
					abs_idx = all_sessions.index(sess)
				else:
					abs_idx = rel_idx_int + fallback_offset
				out[abs_idx] = trial_data
		return out

	train_trials = remap(train_trials_raw, training_days, fallback_offset=0)
	val_trials = remap(val_trials_raw, target_days, fallback_offset=len(training_days))
	test_trials = remap(test_trials_raw, test_days, fallback_offset=len(training_days) + len(target_days))
	competition_trials = remap(competition_trials_raw, target_days, fallback_offset=len(training_days))

	return {
		"train": train_trials,
		"val": val_trials,
		"test": test_trials,
		"competition": competition_trials,
		"training_days": training_days,
		"target_days": target_days,
		"manifest_path": manifest_path,
	}


def _filter_trials_to_sessions(trials: Dict[int, object], sessions_eval: List[str], all_sessions: List[str]) -> Dict[int, object]:
	"""
	Keep only day indices whose session name is in sessions_eval.
	(sessions_eval are strings like 't15.2023.10.01')
	"""
	allowed = set(sessions_eval)
	out = {}
	for day_idx, trial_data in trials.items():
		if 0 <= int(day_idx) < len(all_sessions) and all_sessions[int(day_idx)] in allowed:
			out[int(day_idx)] = trial_data
	return out


# ---------------------------------------------------------------------------
# LM decoding (T12 format)
# ---------------------------------------------------------------------------
def _ensure_lm_decoder():
	global ngramDecoder, LM_CAN_RESCORE
	if ngramDecoder is None and LOAD_LM:
		import lm_utils as lmDecoderUtils
		lm_dir = _resolve_lm_dir()
		LM_CAN_RESCORE = (
			os.path.exists(os.path.join(lm_dir, "G.fst"))
			and os.path.exists(os.path.join(lm_dir, "G_no_prune.fst"))
		)
		if RESCORE and not LM_CAN_RESCORE:
			print("[LM] RESCORE requested but G_no_prune.fst is missing; disabling rescoring for stability.")
		ngramDecoder = lmDecoderUtils.build_lm_decoder(
			lm_dir,
			acoustic_scale=ACOUSTIC_SCALE,
			nbest=N_BEST,
			beam=LM_BEAM,
		)
	return ngramDecoder


def get_lm_outputs(tf_logits: torch.Tensor, n_frames: Optional[int] = None, debug_label: Optional[str] = None):
	import lm_utils as lmDecoderUtils
	from tta_utils import clean_transcription, get_phonemes

	decoder = _ensure_lm_decoder()

	# Decode exactly one utterance at a time. Trim to valid frames and
	# enforce float32/contiguous layout for the native decoder.
	if tf_logits.ndim == 3:
		logits_np = tf_logits[0].detach().cpu().float().numpy()
	elif tf_logits.ndim == 2:
		logits_np = tf_logits.detach().cpu().float().numpy()
	else:
		raise ValueError(f"Expected logits rank 2 or 3, got shape {tuple(tf_logits.shape)}")

	if n_frames is not None:
		n_frames = int(max(1, n_frames))
		logits_np = logits_np[:n_frames, :]

	logits_np = logits_np[None, :, :]
	logits_np = np.concatenate([logits_np[:, :, 1:], logits_np[:, :, 0:1]], axis=-1)  # blank to last
	logits_np = lmDecoderUtils.rearrange_speech_logits(logits_np, has_sil=True)
	logits_np = np.ascontiguousarray(np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
	if debug_label is not None:
		h = hashlib.md5(logits_np.tobytes()).hexdigest()[:10]
		print(f"[dbg] lm input {debug_label} md5={h} shape={logits_np.shape}", flush=True)
	decoded = lmDecoderUtils.lm_decode(
		decoder,
		logits_np[0],
		blankPenalty=BLANK_PENALTY,
		returnNBest=RETURN_N_BEST,
		rescore=bool(RESCORE and LM_CAN_RESCORE),
	)

	decoded = clean_transcription(decoded)
	y_pseudo, y_len_pseudo = get_phonemes(decoded)
	return decoded, y_pseudo, y_len_pseudo


# ---------------------------------------------------------------------------
# Data loading (MATCHES your trainer val_loader exactly)
# ---------------------------------------------------------------------------
def _make_eval_loader_from_manifest(args_yaml: dict, sessions_eval: List[str], split_name: str):
	"""
	split_name:
	  - "val"  -> use manifest['val'] (val_trial_indicies remapped)
	  - "test" -> use manifest['test'] (test_trial_indicies remapped). Competition is not used here.
	MATCHES trainer snippet:
	  BrainToTextDataset(..., split='test', days_per_batch=None, n_batches=None, shuffle=False, num_workers=0)
	"""
	from torch.utils.data import DataLoader
	try:
		from model_training.dataset import BrainToTextDataset
	except ImportError:
		from dataset import BrainToTextDataset

	all_sessions = args_yaml["dataset"]["sessions"]
	man = _load_manifest_trials(args_yaml)

	if split_name == "val":
		trials_abs = man["val"]
	elif split_name == "test":
		trials_abs = man["test"]
	else:
		raise ValueError(f"split_name must be 'val' or 'test', got {split_name}")

	trials_abs = _filter_trials_to_sessions(trials_abs, sessions_eval, all_sessions)

	# This evaluation/TTA path decodes one utterance at a time.
	# For safety and correctness, force single-trial batches.
	batch_size = 1
	feature_subset = _get_from_nested(args_yaml, ["dataset", "feature_subset"], None)

	ds = BrainToTextDataset(
		trial_indicies=trials_abs,
		split="test",            # EXACTLY like trainer val_dataset
		days_per_batch=None,
		n_batches=None,
		batch_size=batch_size,
		must_include_days=None,
		random_seed=int(_get_from_nested(args_yaml, ["dataset", "seed"], 0)),
		feature_subset=feature_subset,
	)

	loader = DataLoader(
		ds,
		batch_size=None,
		shuffle=False,           # EXACTLY like trainer val_loader
		num_workers=0,           # EXACTLY like trainer val_loader
		pin_memory=True,
	)
	return loader


# ---------------------------------------------------------------------------
# Model loading (GRUDecoder)
# ---------------------------------------------------------------------------
def _build_gru_from_args(args_yaml: dict, device: torch.device):
	try:
		from model_training.rnn_model import GRUDecoder
	except ImportError:
		from rnn_model import GRUDecoder

	sessions = args_yaml["dataset"]["sessions"]
	n_days = len(sessions)
	n_classes = int(args_yaml["dataset"]["n_classes"])

	model = GRUDecoder(
		neural_dim=int(args_yaml["model"]["n_input_features"]),
		n_units=int(args_yaml["model"]["n_units"]),
		n_days=n_days,
		n_classes=n_classes,
		rnn_dropout=float(args_yaml["model"]["rnn_dropout"]),
		input_dropout=float(args_yaml["model"]["input_network"]["input_layer_dropout"]),
		n_layers=int(args_yaml["model"]["n_layers"]),
		patch_size=int(args_yaml["model"]["patch_size"]),
		patch_stride=int(args_yaml["model"]["patch_stride"]),
	).to(device)
	return model

def _build_gru_mdan_from_args(args_yaml: dict, device: torch.device):
	try:
		from model_training.rnn_model_mdan import GRUDecoder as GRUDecoderMDAN
	except ImportError:
		from rnn_model_mdan import GRUDecoder as GRUDecoderMDAN

	sessions = args_yaml["dataset"]["sessions"]
	n_days = len(sessions)
	n_classes = int(args_yaml["dataset"]["n_classes"])

	model = GRUDecoderMDAN(
		neural_dim=int(args_yaml["model"]["n_input_features"]),
		n_units=int(args_yaml["model"]["n_units"]),
		n_days=n_days,
		n_classes=n_classes,
		rnn_dropout=float(args_yaml["model"]["rnn_dropout"]),
		input_dropout=float(args_yaml["model"]["input_network"]["input_layer_dropout"]),
		n_layers=int(args_yaml["model"]["n_layers"]),
		patch_size=int(args_yaml["model"]["patch_size"]),
		patch_stride=int(args_yaml["model"]["patch_stride"]),
	).to(device)
	return model


def _set_eval_dropouts_to_zero(model: torch.nn.Module):
	for m in model.modules():
		if isinstance(m, torch.nn.Dropout):
			m.p = 0.0


def _freeze_for_tta(model: torch.nn.Module):
	"""Freeze for TTA: unfreeze only day layers (day_weights, day_biases in baseline/MDAN).
	Do not unfreeze the final out layer."""
	for name, p in model.named_parameters():
		trainable = ("day_" in name) or (name in {"dayWeights", "dayBias"})
		p.requires_grad = bool(trainable)
	if not any(p.requires_grad for p in model.parameters()):
		print("[warn] no trainable params matched TTA freeze; unfreezing all for TTA.")
		for p in model.parameters():
			p.requires_grad = True


# Alias so notebooks/callers can use either name after re-import.
_freeze_for_tta_day_only = _freeze_for_tta


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------
def _ctc_loss_exact(logits, labels, adjusted_lens, phone_seq_lens):
	ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
	task_loss = ctc(
		log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
		targets=labels,
		input_lengths=adjusted_lens,
		target_lengths=phone_seq_lens,
	)
	return torch.mean(task_loss)


def run_tta_eval_gru_vs_ours_t12format(
	*,
	device: torch.device,
	split_name: str,                 # "val" or "test"
	sessions_eval: List[str],
	day_deltas: List[int],
	last_day_idx: int,
	last_val_day_idx: int,
	seed: int,
	tta_mode: str,                   # "baseline" or "corp"
	ours_lr: Optional[float],
	baseline_checkpoints_dir: str,
	ours_checkpoints_dir: str,
	baseline_eval_args: dict,
	corp_args_gru: dict,
	init_from_ckpt: Optional[dict] = None,   # {"gru": "...pth", "ours": "...pth"}
	save_tta_ckpt_dir: Optional[str] = None,
	run_gru: bool = True,
	run_dann: bool = True,
	wer_debug: bool = False,
):
	from lm_utils import _cer_and_wer
	from tta_utils import clean_transcription

	out = {
		"split": split_name,
		"tta_mode": tta_mode,
		"seed": seed,
		"last_day_idx": last_day_idx,
		"last_val_day_idx": last_val_day_idx,
		"day_deltas": day_deltas,
		"gru": None,
		"dann": None,
	}

	base_ckpt_dir = Path(baseline_checkpoints_dir)
	ours_ckpt_dir = Path(ours_checkpoints_dir)

	args_base = _load_yaml(str(base_ckpt_dir / "args.yaml"))
	args_ours = _load_yaml(str(ours_ckpt_dir / "args.yaml"))

	# enforce sessions order used for indexing
	args_base.setdefault("dataset", {})
	args_ours.setdefault("dataset", {})
	args_base["dataset"]["sessions"] = SESSIONS_26
	args_ours["dataset"]["sessions"] = SESSIONS_26

	model_gru = _build_gru_from_args(args_base, device)
	model_ours = _build_gru_mdan_from_args(args_ours, device)

	if init_from_ckpt is not None and init_from_ckpt.get("gru"):
		ckpt_gru = Path(init_from_ckpt["gru"])
	else:
		ckpt_gru = _resolve_best_checkpoint(base_ckpt_dir)

	if init_from_ckpt is not None and init_from_ckpt.get("ours"):
		ckpt_ours = Path(init_from_ckpt["ours"])
	else:
		ckpt_ours = _resolve_best_checkpoint(ours_ckpt_dir)

	ckpt_gru_resolved = Path(str(ckpt_gru)).resolve()
	ckpt_ours_resolved = Path(str(ckpt_ours)).resolve()
	if ckpt_gru_resolved == ckpt_ours_resolved:
		raise RuntimeError(
			f"gru and ours must load different checkpoints; both resolve to {ckpt_gru_resolved}. "
			"Check ours_dir and that ours_dir/best_checkpoint is not a symlink to baseline."
		)
	_load_ckpt_state(model_gru, str(ckpt_gru), device, fill_missing_day_from_idx=last_day_idx)
	_load_ckpt_state(model_ours, str(ckpt_ours), device, fill_missing_day_from_idx=last_val_day_idx)
	print(f"[ckpt] gru weights: {ckpt_gru}", flush=True)
	print(f"[ckpt] ours weights: {ckpt_ours}", flush=True)
	# Debug: print val_PER / val_loss from checkpoint (same format as rnn_trainer_baseline.save_model_checkpoint)
	def _ckpt_metrics(path: Path) -> dict:
		try:
			ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
			if not isinstance(ckpt, dict):
				return {"val_PER": None, "val_loss": None}
			val_PER = ckpt.get("val_PER")
			val_loss = ckpt.get("val_loss")
			return {"val_PER": val_PER, "val_loss": val_loss}
		except Exception:
			return {"val_PER": None, "val_loss": None}
	for label, path in [("gru", ckpt_gru), ("ours", ckpt_ours)]:
		m = _ckpt_metrics(Path(path))
		if m["val_PER"] is not None:
			p = m["val_PER"]
			l = m["val_loss"]
			print(f"[ckpt] {label} val_PER={p:.4f}" if isinstance(p, (int, float)) else f"[ckpt] {label} val_PER={p}", flush=True)
			if l is not None:
				print(f"[ckpt] {label} val_loss={l:.6f}" if isinstance(l, (int, float)) else f"[ckpt] {label} val_loss={l}", flush=True)
	# Sanity: confirm the two models have different weights (same => bug)
	def _param_fingerprint(m, name_prefix, max_params=3):
		out = []
		for n, p in m.named_parameters():
			if n.startswith(name_prefix) and p.numel() > 0:
				out.append(float(p.flatten()[: min(5, p.numel())].sum().item()))
				if len(out) >= max_params:
					break
		return tuple(out)
	# Baseline: gru.* ; MDAN: gru_layers.*
	fp_gru = _param_fingerprint(model_gru, "gru.")
	fp_ours = _param_fingerprint(model_ours, "gru_layers.")
	if fp_gru == fp_ours and len(fp_gru) > 0:
		print(f"[WARN] gru and ours parameter fingerprints match {fp_gru}; check that different checkpoints are loaded.", flush=True)
	else:
		print(f"[ckpt] gru fingerprint: {fp_gru[:2]}... ours fingerprint: {fp_ours[:2]}...", flush=True)

	# MANIFEST-ONLY eval loaders (matches trainer val_loader style)
	loader_gru = _make_eval_loader_from_manifest(args_base, sessions_eval, split_name=split_name)
	loader_ours = _make_eval_loader_from_manifest(args_ours, sessions_eval, split_name=split_name)

	# TTA LRs: match notebook (GRU 0.02, ours 0.03)
	GRU_TTA_LR = 0.02
	OURS_TTA_LR = 0.03

	def _eval_one(model: torch.nn.Module, model_key: str, effective_args: dict, loader):
		n_trials = len(loader)
		print(f"[eval] {split_name} {model_key}: {n_trials} trials", flush=True)
		model.eval()
		_set_eval_dropouts_to_zero(model)

		optimizer = None
		model_frozen = None
		trainable_params = None
		if tta_mode != "baseline":
			assert (ours_lr is not None) or (model_key == "gru"), "ours_lr must be set for TTA runs"
			lr_this = GRU_TTA_LR if model_key == "gru" else (float(ours_lr) if ours_lr is not None else OURS_TTA_LR)
			_freeze_for_tta(model)
			trainable_params = [p for p in model.parameters() if p.requires_grad]
			optimizer = torch.optim.AdamW(
				trainable_params,
				lr=lr_this,
				weight_decay=float(effective_args["l2_decay"]),
				eps=0.1,
				betas=(0.9, 0.999),
			)
			model_frozen = copy.deepcopy(model).to(device)
			model_frozen.eval()
			for p in model_frozen.parameters():
				p.requires_grad_(False)

		decoded_list_all = []
		transcripts_all = []
		per_day_wer = []

		# Always use last train day for both val and test (both models).
		forced_day_idx = int(last_day_idx)
		forced_day = torch.tensor([forced_day_idx], dtype=torch.int64, device=device)

		cur_day = None
		cur_decoded = []
		cur_trans = []

		def _flush_day():
			nonlocal cur_decoded, cur_trans, per_day_wer, decoded_list_all, transcripts_all, cur_day
			if cur_day is None:
				return
			_, wer = _cer_and_wer(cur_decoded, cur_trans, outputType="speech", returnCI=False)
			per_day_wer.append(float(wer))
			idx = len(per_day_wer) - 1
			delta_str = str(day_deltas[idx]) if idx < len(day_deltas) else "?"
			print(f"[per-day WER] {split_name} {model_key} day_delta={delta_str} (day_idx={cur_day}) WER={wer:.2%} n={len(cur_decoded)}", flush=True)
			decoded_list_all.extend(cur_decoded)
			transcripts_all.extend(cur_trans)
			cur_decoded = []
			cur_trans = []

		patch_size = int(_get_from_nested(args_base if model_key == "gru" else args_ours, ["model", "patch_size"], 0))
		patch_stride = int(_get_from_nested(args_base if model_key == "gru" else args_ours, ["model", "patch_stride"], 1))

		total_trials = len(loader)
		for batch_idx, batch in enumerate(loader):
			features = batch["input_features"].to(device)
			labels = batch["seq_class_ids"].to(device)
			n_time_steps = batch["n_time_steps"].to(device)
			phone_seq_lens = batch["phone_seq_lens"].to(device)
			day_indicies = batch["day_indicies"].to(device)

			day_val = int(day_indicies[0].item())
			if cur_day is None:
				cur_day = day_val
			elif day_val != cur_day:
				_flush_day()
				cur_day = day_val

			adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
			B = int(features.shape[0])
			day_idx = forced_day.expand(B)

			if tta_mode == "baseline":
				with torch.no_grad():
					logits_eval = model(features, day_idx if B > 1 else forced_day)
					T_logits = int(logits_eval.shape[1])
					n_frames_used = T_logits
					if wer_debug and batch_idx < 2:
						print(f"[dbg] {split_name} {model_key} batch={batch_idx} T_logits={T_logits}", flush=True)
					decoded, y_pseudo, y_len_pseudo = get_lm_outputs(
						logits_eval, n_frames=n_frames_used,
						debug_label=f"{split_name}_{model_key}_b{batch_idx}" if (wer_debug and batch_idx < 2) else None,
					)
			else:
				# TTA: PRE from frozen model (matches no-TTA for every batch, same as notebook)
				with torch.no_grad():
					logits_pre = model_frozen(features, day_idx if B > 1 else forced_day)
					T_pre = int(logits_pre.shape[1])
					decoded_pre, y_pseudo, y_len_pseudo = get_lm_outputs(
						logits_pre, n_frames=T_pre,
						debug_label=f"{split_name}_{model_key}_b{batch_idx}" if (wer_debug and batch_idx < 2) else None,
					)
					if wer_debug and batch_idx < 2:
						print(f"[dbg] decoded PRE {model_key}: {repr(decoded_pre)[:80]}", flush=True)

				n_augs = int(effective_args["repeats"][0])
				X = features
				y_p = y_pseudo.to(device)
				ypl = y_len_pseudo.to(device) if isinstance(y_len_pseudo, torch.Tensor) else torch.tensor([int(y_len_pseudo)], device=device, dtype=torch.int32)

				if n_augs > 1:
					X = X.repeat(n_augs, 1, 1)
					day_rep = forced_day.expand(X.shape[0])
					y_p = y_p.unsqueeze(0).repeat(n_augs, 1)
					ypl = ypl.repeat(n_augs)
				else:
					day_rep = day_idx if B > 1 else forced_day

				if bool(effective_args.get("WN+BS", False)) and n_augs > 1:
					X = X + torch.randn_like(X) * float(effective_args["white_noise"])
					X = X + torch.randn((X.shape[0], 1, X.shape[2]), device=device) * float(effective_args["baseline_shift"])

				model.train()
				for _ in range(int(effective_args["adaptation_steps"])):
					logits = model(X, day_rep if X.shape[0] != B else (day_idx if B > 1 else forced_day))
					T_logits = int(logits.shape[1])
					lens = torch.full((logits.shape[0],), T_logits, device=device, dtype=torch.int32)
					loss = _ctc_loss_exact(logits, y_p, lens, ypl)
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
					optimizer.step()

				model.eval()
				_set_eval_dropouts_to_zero(model)
				with torch.no_grad():
					logits_post = model(features, day_idx if B > 1 else forced_day)
					T2 = int(logits_post.shape[1])
					decoded, _, _ = get_lm_outputs(logits_post, n_frames=T2)

			cur_decoded.append(decoded)
			cur_trans.append(clean_transcription(_transcription_tensor_to_str(batch["transcriptions"][0])))

		_flush_day()
		_, wer_total = _cer_and_wer(decoded_list_all, transcripts_all, outputType="speech", returnCI=False)
		print(f"[TOTAL WER] {split_name} {model_key}: {wer_total:.2%}", flush=True)

		tta_ckpt_path = None
		if save_tta_ckpt_dir is not None and tta_mode != "baseline":
			os.makedirs(save_tta_ckpt_dir, exist_ok=True)
			tta_ckpt_path = os.path.join(save_tta_ckpt_dir, f"{model_key}_posttta.pth")
			torch.save(model.state_dict(), tta_ckpt_path)
			print(f"[ckpt] saved: {tta_ckpt_path}", flush=True)

		return {"per_day_wer": per_day_wer, "wer_total": float(wer_total), "tta_ckpt_path": tta_ckpt_path}

	eff = dict(baseline_eval_args) if tta_mode == "baseline" else dict(corp_args_gru)

	if run_gru:
		out["gru"] = _eval_one(model_gru, "gru", eff, loader_gru)
	if run_dann:
		out["dann"] = _eval_one(model_ours, "ours", eff, loader_ours)
	return out


# ---------------------------------------------------------------------------
# Plotting (same expectations)
# ---------------------------------------------------------------------------
def plot_wer_val_test_tta(
	*,
	out_val_set_with_tta=None,
	eval_day_deltas=None,
	out_val_set_without_tta=None,
	out_unseen_test_set_with_tta=None,
	test_day_deltas=None,
	out_unseen_test_set_without_tta=None,
	dataset_name="Val/Test",
	title_extra=None,
	figsize=(11, 8),
	val_linestyle="-",
	test_linestyle="-",
	val_marker="o",
	test_marker="o",
	show_means=True,
	save_path: str | None = None,
):
	import matplotlib.pyplot as plt
	import matplotlib.colors as mcolors

	fig, ax = plt.subplots(figsize=figsize)

	COLOR = {"gru": mcolors.to_rgb("tab:green"), "dann": mcolors.to_rgb("tab:orange")}
	LABEL = {"gru": "GRU", "dann": "Ours"}

	def _extract(run):
		if run is None:
			return {}
		out = {}
		for k in ("gru", "dann"):
			if run.get(k) is not None and run[k].get("per_day_wer") is not None:
				out[k] = {
					"per_day_wer": list(run[k]["per_day_wer"]),
					"wer_total": run[k].get("wer_total", None),
				}
		return out

	def _plot(run_or_runs, x, split_name, tta_name, linestyle, marker):
		if run_or_runs is None or x is None:
			return

		runs = run_or_runs if isinstance(run_or_runs, list) else [run_or_runs]
		series_list = [_extract(r) for r in runs]
		series_list = [s for s in series_list if len(s) > 0]
		if not series_list:
			return

		for k in ("gru", "dann"):
			all_pd = []
			all_mt = []
			for s in series_list:
				if k in s:
					all_pd.append(np.array(s[k]["per_day_wer"], dtype=float))
					if s[k]["wer_total"] is not None:
						all_mt.append(float(s[k]["wer_total"]))

			if not all_pd:
				continue

			max_len = max(len(a) for a in all_pd)
			pd_stack = []
			for a in all_pd:
				if len(a) < max_len:
					a = np.concatenate([a, np.repeat(a[-1], max_len - len(a))])
				pd_stack.append(a)
			pd_stack = np.stack(pd_stack, axis=0)

			pd_mean = pd_stack.mean(axis=0)
			pd_std = pd_stack.std(axis=0) if len(all_pd) > 1 else None
			mt_mean = np.mean(all_mt) if all_mt else None

			base_rgb = COLOR[k]
			rgb = _lighten_rgb(base_rgb, 0.55) if tta_name == "no TTA" else base_rgb

			ax.plot(
				x,
				pd_mean,
				linestyle=linestyle,
				marker=marker,
				color=rgb,
				alpha=1.0,
				linewidth=3,
				markersize=14,
				markeredgewidth=0,
				label=f"{LABEL[k]} {split_name} ({tta_name})",
			)
			if pd_std is not None:
				ax.fill_between(x, pd_mean - pd_std, pd_mean + pd_std, color=rgb, alpha=0.2)

			if show_means and mt_mean is not None:
				ax.axhline(
					mt_mean,
					linestyle="-",
					color=rgb,
					alpha=0.25,
					label=f"{LABEL[k]} {split_name} mean ({tta_name}): {mt_mean:.4f}",
				)

	if out_val_set_without_tta is not None:
		_plot(out_val_set_without_tta, eval_day_deltas, "Val", "no TTA", val_linestyle, val_marker)
	if out_val_set_with_tta is not None:
		_plot(out_val_set_with_tta, eval_day_deltas, "Val", "TTA", val_linestyle, val_marker)

	if out_unseen_test_set_without_tta is not None:
		_plot(out_unseen_test_set_without_tta, test_day_deltas, "Test", "no TTA", test_linestyle, test_marker)
	if out_unseen_test_set_with_tta is not None:
		_plot(out_unseen_test_set_with_tta, test_day_deltas, "Test", "TTA", test_linestyle, test_marker)

	title = f"Per-Day WER ({dataset_name})"
	if title_extra:
		title += f"\n{title_extra}"

	has_multiple_runs = any(
		isinstance(x, list) and len(x) > 1
		for x in (
			out_val_set_without_tta,
			out_val_set_with_tta,
			out_unseen_test_set_without_tta,
			out_unseen_test_set_with_tta,
		)
		if x is not None
	)
	if has_multiple_runs:
		title += " (mean ± std)"
	ax.set_title(title, fontsize=13, fontweight="bold")

	ax.set_xlabel("Days Away from Last Trained Day")
	ax.set_ylabel("Word Error Rate (WER)")
	ax.grid(False)
	ax.set_ylim(0.10, 1.05)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	xticks = set()
	if eval_day_deltas is not None:
		xticks |= set(eval_day_deltas)
	if test_day_deltas is not None:
		xticks |= set(test_day_deltas)
	if xticks:
		xticks = sorted(xticks)
		ax.set_xticks(xticks)
		ax.set_xticklabels([str(x) for x in xticks])

	handles, labels = ax.get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc="lower center",
		bbox_to_anchor=(0.5, -0.02),
		ncol=2,
		fontsize=9,
		frameon=True,
	)
	fig.subplots_adjust(bottom=0.25)

	if save_path is not None:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		fig.savefig(save_path, bbox_inches="tight")
		pdf_path = str(save_path).rsplit(".", 1)[0] + ".pdf" if "." in save_path else save_path + ".pdf"
		fig.savefig(pdf_path, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()

	return fig, ax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser("T15 n-gram WER eval in T12 format (manifest-only)")
	parser.add_argument("--split", choices=["split1", "split2"], required=True)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--results-dir", type=str, default="/victoriapvc/results/wer_t15_new")
	parser.add_argument("--n-runs", type=int, default=5)
	parser.add_argument("--ours-tta-lrs", type=float, nargs="+", default=[0.03],
					help="TTA learning rates for ours model (default: 0.03). GRU uses 0.02.")
	parser.add_argument("--skip-tta-compute", action="store_true")
	parser.add_argument("--replot-only", action="store_true")
	parser.add_argument("--no-cache", action="store_true", help="Ignore all cache; recompute gru and ours for every run (for validation)")
	parser.add_argument("--wer-debug", action="store_true", help="Print n_frames, logits stats, nonfinite, decoded sample and lm input hash (first batch per model)")
	args = parser.parse_args()

	# Headless-safe plotting (no DISPLAY required)
	os.environ.setdefault("MPLBACKEND", "Agg")

	device = torch.device(args.device)

	cfg = split_config(args.split)
	if args.split == "split1":
		last_day_idx = 4
		last_val_day_idx = 17
	elif args.split == "split2":
		last_day_idx = 9
		last_val_day_idx = 17
	elif args.split == "split3":
		last_day_idx = 4
		last_val_day_idx = 9
	elif args.split == "split5":
		last_day_idx = 9
		last_val_day_idx = 14
	else:
		raise ValueError(f"Unknown split: {args.split}")
		
	last_val_day_idx = ev.SESSIONS_26.index(val_sessions[0])  # day index for "ours" model
	train_sessions = list(cfg["train_sess"])
	val_sessions = list(cfg["val_sess"])
	test_sessions = list(cfg["test_sess"])

	baseline_dir = Path(cfg["baseline_dir"])
	ours_dir = Path(cfg["ours_dir"])
	baseline_ckpt_dir = baseline_dir / "checkpoints"
	ours_ckpt_dir = ours_dir
	if not (baseline_ckpt_dir / "args.yaml").exists():
		raise FileNotFoundError(f"Missing baseline args.yaml: {baseline_ckpt_dir / 'args.yaml'}")
	if not (ours_ckpt_dir / "args.yaml").exists():
		raise FileNotFoundError(f"Missing ours args.yaml: {ours_ckpt_dir / 'args.yaml'}")
	print(f"[ckpt] baseline checkpoint dir: {baseline_ckpt_dir}")
	print(f"[ckpt] ours checkpoint dir: {ours_ckpt_dir}")

	# compute day deltas
	last_train_date = _parse_t15_date(train_sessions[-1])
	eval_day_deltas = [(_parse_t15_date(s) - last_train_date).days for s in val_sessions]
	test_day_deltas = [(_parse_t15_date(s) - last_train_date).days for s in test_sessions]

	if args.split == "split1":
		n_train_days = 5
		n_val_days = 13
		n_test_days = 8
		
	elif args.split == "split2":
		n_train_days = 10
		n_val_days = 8
		n_test_days = 8

	baseline_eval_args = {
		"dropout": 0.0,
		"input_dropout": 0.0,
		"max_mask_pct": 0.0,
		"num_masks": 0,
		"gru": True,
		"max_day": n_train_days + n_val_days - 1,
		"repeats": [1],
		"l2_decay": 0.0,
	}

	# TTA hyperparams: match notebook_wer_val_day1_debug.ipynb tta_args
	corp_args_gru = {
		"learning_rate": None,
		"repeats": [64],
		"adaptation_steps": 1,
		"WN+BS": True,
		"white_noise": 1.0,
		"baseline_shift": 0.2,
		"dropout": 0.4,
		"input_dropout": 0.0,
		"l2_decay": 0.001,
		"max_mask_pct": 0.0,
		"num_masks": 0,
		"freeze_patch": False,
		"freeze_linear": True,
		"freeze_up_to_rep_layer_idx": False,
		"gru": True,
		"max_day": n_train_days + n_val_days - 1,
	}

	results_root = Path(args.results_dir) / cfg["data_subdir"]
	results_root.mkdir(parents=True, exist_ok=True)
	cache_dir = results_root / "cache"
	cache_dir.mkdir(parents=True, exist_ok=True)

	timestamp = time.strftime("%Y%m%d_%H%M%S")

	global ngramDecoder
	ngramDecoder = None
	print(f"[LM] T15 setup: beam={LM_BEAM}, acoustic_scale={ACOUSTIC_SCALE}, blank_penalty=log(9)")

	def _get_or_run(*, split_name: str, tta_mode: str, seed: int, ours_lr: Optional[float],
					init_from_ckpt: Optional[dict] = None, save_tta_dir: Optional[str] = None):
		init_sig = None
		if init_from_ckpt:
			init_sig = tuple(sorted((str(k), str(v)) for k, v in init_from_ckpt.items()))
		extra_key = {
			"split": args.split,
			"last_day_idx": last_day_idx,
			"last_val_day_idx": last_val_day_idx,
			"baseline_dir": str(baseline_dir),
			"ours_dir": str(ours_dir),
			"lm": (LM_BEAM, ACOUSTIC_SCALE, BLANK_PENALTY),
			"val_sessions": tuple(val_sessions),
			"test_sessions": tuple(test_sessions),
			"tta_mode": tta_mode,
			"init_from_ckpt": init_sig,
		}
		# Val: try both True and False so we hit cache from either old (test=competition) or intermediate runs.
		# Test: use False so cache key differs from old runs and test re-computes on manifest["test"].
		extra_key["manifest_policy_test_is_competition"] = (split_name != "test")
		cpath = _run_cache_path(cache_dir, split_name, tta_mode, seed, ours_lr, extra_key)

		if args.replot_only:
			cached = _load_cached_run(cpath)
			if cached is None:
				raise FileNotFoundError(f"Missing cache for replot-only: {cpath}")
			return cached

		cached = None
		if not getattr(args, "no_cache", False):
			cached = _load_cached_run(cpath)
		if cached is None and split_name == "val" and not getattr(args, "no_cache", False):
			extra_key_alt = dict(extra_key)
			extra_key_alt["manifest_policy_test_is_competition"] = False
			cpath_alt = _run_cache_path(cache_dir, split_name, tta_mode, seed, ours_lr, extra_key_alt)
			cached = _load_cached_run(cpath_alt)
			if cached is not None:
				cpath = cpath_alt
		if cached is None and split_name == "val" and not getattr(args, "no_cache", False):
			cpath_any = _find_any_val_cache(cache_dir, split_name, tta_mode, seed, ours_lr)
			if cpath_any is not None:
				cached = _load_cached_run(cpath_any)
				if cached is not None:
					cpath = cpath_any
		if cached is not None:
			# Val and test: keep cached gru, but always recompute ours (dann) so MDAN/checkpoint changes are picked up.
			if cached.get("gru") is not None:
				print(f"[cache] hit (gru): {cpath}; recomputing ours (dann) only", flush=True)
				sessions_eval = val_sessions if split_name == "val" else test_sessions
				deltas = eval_day_deltas if split_name == "val" else test_day_deltas
				partial = run_tta_eval_gru_vs_ours_t12format(
					device=device,
					split_name=split_name,
					sessions_eval=sessions_eval,
					day_deltas=deltas,
					last_day_idx=last_day_idx,
					last_val_day_idx=last_val_day_idx,
					seed=seed,
					tta_mode=tta_mode,
					ours_lr=ours_lr,
					baseline_checkpoints_dir=str(baseline_ckpt_dir),
					ours_checkpoints_dir=str(ours_ckpt_dir),
					baseline_eval_args=baseline_eval_args,
					corp_args_gru=corp_args_gru,
					init_from_ckpt=init_from_ckpt,
					save_tta_ckpt_dir=save_tta_dir,
					run_gru=False,
					run_dann=True,
					wer_debug=getattr(args, "wer_debug", False),
				)
				merged = dict(cached)
				merged["dann"] = partial["dann"]
				_save_cached_run(cpath, merged)
				return merged
			print(f"[cache] hit: {cpath}", flush=True)
			return cached

		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

		sessions_eval = val_sessions if split_name == "val" else test_sessions
		deltas = eval_day_deltas if split_name == "val" else test_day_deltas

		out = run_tta_eval_gru_vs_ours_t12format(
			device=device,
			split_name=split_name,
			sessions_eval=sessions_eval,
			day_deltas=deltas,
			last_day_idx=last_day_idx,
			last_val_day_idx=last_val_day_idx,
			seed=seed,
			tta_mode=tta_mode,
			ours_lr=ours_lr,
			baseline_checkpoints_dir=str(baseline_ckpt_dir),
			ours_checkpoints_dir=str(ours_ckpt_dir),
			baseline_eval_args=baseline_eval_args,
			corp_args_gru=corp_args_gru,
			init_from_ckpt=init_from_ckpt,
			save_tta_ckpt_dir=save_tta_dir,
			wer_debug=getattr(args, "wer_debug", False),
		)
		_save_cached_run(cpath, out)
		return out

	def _collect_runs(split_name: str, tta_mode: str, ours_lr: Optional[float],
					  init_from_ckpt: Optional[dict] = None, save_dir: Optional[Path] = None,
					  n_runs: Optional[int] = None):
		n = int(n_runs) if n_runs is not None else int(args.n_runs)
		outs = []
		for run_idx in range(n):
			seed = 42 + run_idx
			save_dir_str = None
			if save_dir is not None:
				save_dir_str = str(save_dir / f"seed{seed}")
			outs.append(_get_or_run(
				split_name=split_name,
				tta_mode=tta_mode,
				seed=seed,
				ours_lr=ours_lr,
				init_from_ckpt=init_from_ckpt,
				save_tta_dir=save_dir_str
			))
		return outs

	# Eval policy: val days -> manifest["val"] (train+val in val sessions); test days -> manifest["test"] (train+val in test sessions).
	# No TTA: 1 run per model. TTA: n_runs (e.g. 5) seeds. Then plot all.
	# 1) no-TTA (single run per split)
	print("\n=== NO-TTA (Val/Test) — 1 run per model ===")
	val_no_tta_runs = _collect_runs("val", "baseline", ours_lr=None, n_runs=1)
	test_no_tta_runs = _collect_runs("test", "baseline", ours_lr=None, n_runs=1)

	no_tta_pkl_path = results_root / f"no_tta_{timestamp}.pkl"
	with open(no_tta_pkl_path, "wb") as f:
		pickle.dump({
			"split": args.split,
			"val_no_tta": val_no_tta_runs,
			"test_no_tta": test_no_tta_runs,
			"eval_day_deltas": eval_day_deltas,
			"test_day_deltas": test_day_deltas,
			"timestamp": timestamp,
		}, f)
	print(f"[result] saved: {no_tta_pkl_path}", flush=True)

	def _save_no_tta_plots():
		plot_path_no_tta = results_root / f"no_tta_VAL_TEST_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=val_no_tta_runs,
			out_unseen_test_set_without_tta=test_no_tta_runs,
			eval_day_deltas=eval_day_deltas,
			test_day_deltas=test_day_deltas,
			dataset_name=f"T15 {args.split}",
			title_extra="no TTA",
			save_path=str(plot_path_no_tta),
			show_means=True,
		)
		print(f"[plot] finished: {plot_path_no_tta}", flush=True)

		plot_path_no_tta_val = results_root / f"no_tta_VAL_ONLY_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=val_no_tta_runs,
			out_unseen_test_set_without_tta=None,
			eval_day_deltas=eval_day_deltas,
			test_day_deltas=None,
			dataset_name=f"T15 {args.split}",
			title_extra="no TTA (Val only)",
			save_path=str(plot_path_no_tta_val),
			show_means=True,
		)
		print(f"[plot] saved: {plot_path_no_tta_val}")

		plot_path_no_tta_test = results_root / f"no_tta_TEST_ONLY_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=None,
			out_unseen_test_set_without_tta=test_no_tta_runs,
			eval_day_deltas=None,
			test_day_deltas=test_day_deltas,
			dataset_name=f"T15 {args.split}",
			title_extra="no TTA (Test only)",
			save_path=str(plot_path_no_tta_test),
			show_means=True,
		)
		print(f"[plot] finished: {plot_path_no_tta_test}", flush=True)

	try:
		_save_no_tta_plots()
	except Exception as e:
		print(f"[plot] ERROR (no TTA): {e}", flush=True)
		import traceback
		traceback.print_exc()
	print("[LR] finished NO-TTA", flush=True)

	if args.skip_tta_compute:
		print("--skip-tta-compute: skipping all TTA runs.")
		return

	# 2) sweep ours_lr
	for ours_lr in args.ours_tta_lrs:
		print(f"\n=== TTA sweep: ours_lr={ours_lr} ===")

		val_tta_runs = _collect_runs("val", "corp", ours_lr=ours_lr)
		test_tta_runs = _collect_runs("test", "corp", ours_lr=ours_lr)

		# 3) test TTA val-init
		test_tta_valinit_runs = []
		for run_idx in range(int(args.n_runs)):
			seed = 42 + run_idx

			save_dir = results_root / "tta_from_val_ckpts" / f"ourslr{ours_lr:g}" / f"seed{seed}"
			out_val_seed = _get_or_run(
				split_name="val",
				tta_mode="corp",
				seed=seed,
				ours_lr=ours_lr,
				init_from_ckpt=None,
				save_tta_dir=str(save_dir),
			)

			init_from = {}
			if out_val_seed.get("gru") and out_val_seed["gru"].get("tta_ckpt_path"):
				init_from["gru"] = out_val_seed["gru"]["tta_ckpt_path"]
			if out_val_seed.get("dann") and out_val_seed["dann"].get("tta_ckpt_path"):
				init_from["ours"] = out_val_seed["dann"]["tta_ckpt_path"]

			out_test_seed = _get_or_run(
				split_name="test",
				tta_mode="corp",
				seed=seed,
				ours_lr=ours_lr,
				init_from_ckpt=init_from,
				save_tta_dir=None,
			)
			test_tta_valinit_runs.append(out_test_seed)

		agg = {
			"split": args.split,
			"n_runs": int(args.n_runs),
			"ours_lr": float(ours_lr),
			"val_no_tta": val_no_tta_runs,
			"test_no_tta": test_no_tta_runs,
			"val_tta": val_tta_runs,
			"test_tta_no_valinit": test_tta_runs,
			"test_tta_valinit": test_tta_valinit_runs,
			"eval_day_deltas": eval_day_deltas,
			"test_day_deltas": test_day_deltas,
			"lm": {"beam": LM_BEAM, "acoustic_scale": ACOUSTIC_SCALE, "blank_penalty": BLANK_PENALTY},
			"timestamp": timestamp,
		}
		pkl_path = results_root / f"ourslr{ours_lr:g}_{timestamp}.pkl"
		with open(pkl_path, "wb") as f:
			pickle.dump(agg, f)
		print(f"[result] saved: {pkl_path}", flush=True)

		def _print_wer_range(label: str, runs: list):
			if not runs:
				return
			all_wer = []
			for r in runs:
				for k in ("gru", "dann"):
					if r.get(k) and r[k].get("per_day_wer"):
						all_wer.extend(r[k]["per_day_wer"])
					if r.get(k) and r[k].get("wer_total") is not None:
						all_wer.append(r[k]["wer_total"])
			if all_wer:
				arr = np.array(all_wer, dtype=float)
				print(f"[test TTA WER] {label}: min={arr.min():.2%} max={arr.max():.2%} mean={arr.mean():.2%}", flush=True)
		_print_wer_range("no-val-init", test_tta_runs)
		_print_wer_range("val-init", test_tta_valinit_runs)

		plot_path_valinit = results_root / f"ourslr{ours_lr:g}_VALINIT_VAL_TEST_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=val_no_tta_runs,
			out_val_set_with_tta=val_tta_runs,
			out_unseen_test_set_without_tta=test_no_tta_runs,
			out_unseen_test_set_with_tta=test_tta_valinit_runs,
			eval_day_deltas=eval_day_deltas,
			test_day_deltas=test_day_deltas,
			dataset_name=f"T15 {args.split}",
			title_extra="Val-init TTA + no TTA",
			save_path=str(plot_path_valinit),
			show_means=True,
		)
		print(f"[plot] finished: {plot_path_valinit}", flush=True)

		plot_path_novalinit = results_root / f"ourslr{ours_lr:g}_NOVALINIT_VAL_TEST_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=val_no_tta_runs,
			out_val_set_with_tta=val_tta_runs,
			out_unseen_test_set_without_tta=test_no_tta_runs,
			out_unseen_test_set_with_tta=test_tta_runs,
			eval_day_deltas=eval_day_deltas,
			test_day_deltas=test_day_deltas,
			dataset_name=f"T15 {args.split}",
			title_extra="No-val-init TTA + no TTA",
			save_path=str(plot_path_novalinit),
			show_means=True,
		)
		print(f"[plot] finished: {plot_path_novalinit}", flush=True)

		plot_path_novalinit_val = results_root / f"ourslr{ours_lr:g}_NOVALINIT_VAL_ONLY_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=val_no_tta_runs,
			out_val_set_with_tta=val_tta_runs,
			out_unseen_test_set_without_tta=None,
			out_unseen_test_set_with_tta=None,
			eval_day_deltas=eval_day_deltas,
			test_day_deltas=None,
			dataset_name=f"T15 {args.split}",
			title_extra="No-val-init TTA + no TTA (Val only)",
			save_path=str(plot_path_novalinit_val),
			show_means=True,
		)
		print(f"[plot] saved: {plot_path_novalinit_val}")

		plot_path_novalinit_test = results_root / f"ourslr{ours_lr:g}_NOVALINIT_TEST_ONLY_{timestamp}.png"
		plot_wer_val_test_tta(
			out_val_set_without_tta=None,
			out_val_set_with_tta=None,
			out_unseen_test_set_without_tta=test_no_tta_runs,
			out_unseen_test_set_with_tta=test_tta_runs,
			eval_day_deltas=None,
			test_day_deltas=test_day_deltas,
			dataset_name=f"T15 {args.split}",
			title_extra="No-val-init TTA + no TTA (Test only)",
			save_path=str(plot_path_novalinit_test),
			show_means=True,
		)
		print(f"[plot] finished: {plot_path_novalinit_test}", flush=True)
		print(f"[LR] finished ours_lr={ours_lr}", flush=True)


if __name__ == "__main__":
	main()
