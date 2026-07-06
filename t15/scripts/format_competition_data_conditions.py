#!/usr/bin/env python3
"""
Format brain-to-text HDF5 dataset like the transformer script:
- Uses only first 26 sessions.
- Assign sessions to train / target / test by condition.
- For training sessions: if train+val exist, combine; else just train.
- For target sessions: val = train+val, competition = data_test.hdf5.
- For test sessions: test = train+val.

Output:
- output_dir/train/<session>/data.hdf5   (train+val if both exist, else just train)
- output_dir/val/<session>/data.hdf5     (train+val for target sessions)
- output_dir/test/<session>/data.hdf5    (train+val for test sessions)
- output_dir/competition/<session>/data.hdf5  (from data_test.hdf5 for target sessions)
- output_dir/manifest_<suffix>.pkl       (pickle with split paths and metadata)

Conditions (first 26 sessions only):
  1: first 5 = train, next 13 = target/val, last 8 = test  -> output_dir/5train_13val_8test/
  2: first 10 = train, next 8 = target/val, last 8 = test  -> output_dir/10train_8val_8test/
  3: first 5 = train, next 5 = target/val, next 8 = test (first 18 sessions) -> output_dir/5train_5val_8test/
  4: first 5 = train, next 5 = target/val, next 3 = test (first 13 sessions only) -> output_dir/5train_5val_3test/
  5: first 10 = train, next 5 = target/val, next 5 = test (first 20 sessions) -> output_dir/10train_5val_5test/

Usage:
cd /victoriapvc/repos/brain2text-t15

python scripts/format_competition_data_conditions.py \
  --data-dir /victoriapvc/data/hdf5_data_final \
  --output-dir /victoriapvc/data/brain2text_t15_formatted \
  --conditions 5

python scripts/format_competition_data_conditions.py \
  --data-dir /n/holylfs06/LABS/bsabatini_lab/Lab/shunnnli/speechbci/data/T15/hdf5_data_final \
  --output-dir /n/holylfs06/LABS/bsabatini_lab/Lab/shunnnli/speechbci/data/T15/ptDecoder_ctc_both \
"""
import argparse
import os
import pickle
import shutil

import h5py
import numpy as np


# Sessions that have only data_train.hdf5 (no data_val.hdf5); we still include them (train-only source).
SESSIONS_SKIP_NO_VAL_TEST = {
    "t15.2023.08.11",
    "t15.2024.03.03",
    "t15.2024.04.25",
    "t15.2024.04.28",
}

# Default session list for t15 (nejm-brain-to-text); override with --sessions-file or --sessions
DEFAULT_SESSIONS_T15 = [
    "t15.2023.08.11", "t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20",
    "t15.2023.08.25", "t15.2023.08.27", "t15.2023.09.01", "t15.2023.09.03",
    "t15.2023.09.24", "t15.2023.09.29", "t15.2023.10.01", "t15.2023.10.06",
    "t15.2023.10.08", "t15.2023.10.13", "t15.2023.10.15", "t15.2023.10.20",
    "t15.2023.10.22", "t15.2023.11.03", "t15.2023.11.04", "t15.2023.11.17",
    "t15.2023.11.19", "t15.2023.11.26", "t15.2023.12.03", "t15.2023.12.08",
    "t15.2023.12.10", "t15.2023.12.17", "t15.2023.12.29", "t15.2024.02.25",
    "t15.2024.03.03", "t15.2024.03.08", "t15.2024.03.15", "t15.2024.03.17",
    "t15.2024.04.25", "t15.2024.04.28", "t15.2024.05.10", "t15.2024.06.14",
    "t15.2024.07.19", "t15.2024.07.21", "t15.2024.07.28", "t15.2025.01.10",
    "t15.2025.01.12", "t15.2025.03.14", "t15.2025.03.16", "t15.2025.03.30",
    "t15.2025.04.13",
]


def get_trial_indices_from_hdf5(path: str, bad_trials_dict: dict = None, session_name: str = None) -> list:
    """Return list of trial indices (0-based) to include from an HDF5 file, optionally excluding bad trials."""
    if not os.path.exists(path):
        return []
    good = []
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("trial_"))
        for key in keys:
            g = f[key]
            block_num = g.attrs.get("block_num", None)
            trial_num = g.attrs.get("trial_num", None)
            if bad_trials_dict and session_name and block_num is not None and trial_num is not None:
                if (
                    session_name in bad_trials_dict
                    and str(block_num) in bad_trials_dict[session_name]
                    and trial_num in bad_trials_dict[session_name][str(block_num)]
                ):
                    continue
            idx = int(key.replace("trial_", ""))
            good.append(idx)
    return good


def copy_trial(src_f: h5py.File, src_key: str, dst_f: h5py.File, dst_key: str) -> None:
    """Copy one trial group from src to dst (datasets + attributes)."""
    src_g = src_f[src_key]
    dst_f.copy(src_g, dst_f, dst_key)


def combine_and_write_hdf5(
    source_paths: list,
    trial_lists_per_file: list,
    out_path: str,
    session_name: str = None,
) -> int:
    """
    Read trials from multiple source HDF5 files and write them into one output HDF5.
    source_paths: list of paths to source HDF5 files
    trial_lists_per_file: list of lists; trial_lists_per_file[i] = trial indices to take from source_paths[i]
    out_path: output HDF5 path
    Returns total number of trials written.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with h5py.File(out_path, "w") as dst_f:
        for path, trial_list in zip(source_paths, trial_lists_per_file):
            if not trial_list or not os.path.exists(path):
                continue
            with h5py.File(path, "r") as src_f:
                for t in trial_list:
                    src_key = f"trial_{t:04d}"
                    if src_key not in src_f:
                        continue
                    dst_key = f"trial_{written:04d}"
                    copy_trial(src_f, src_key, dst_f, dst_key)
                    written += 1
    return written


def condition_config(condition: int, sessions: list) -> tuple:
    """
    Return (training_names, target_names, test_names, save_suffix).
    Only uses first N sessions (26 for cond 1/2, 18 for cond 3, 13 for cond 4, 20 for cond 5).
    """
    if condition == 1:
        sessions_used = sessions[:26]
        n_train, n_target, n_test = 5, 13, 8
        save_suffix = "_5train_13val_8test"
    elif condition == 2:
        sessions_used = sessions[:26]
        n_train, n_target, n_test = 10, 8, 8
        save_suffix = "_10train_8val_8test"
    elif condition == 3:
        # first 5 train, next 5 val, next 8 test (first 18 sessions)
        sessions_used = sessions[:18]
        n_train, n_target, n_test = 5, 5, 8
        save_suffix = "_5train_5val_8test"
    elif condition == 4:
        # first 13 days only: 5 train, 5 val, 3 test
        sessions_used = sessions[:13]
        n_train, n_target, n_test = 5, 5, 3
        save_suffix = "_5train_5val_3test"
    elif condition == 5:
        # first 20 days only: 10 train, 5 val, 5 test
        sessions_used = sessions[:20]
        n_train, n_target, n_test = 10, 5, 5
        save_suffix = "_10train_5val_5test"
    else:
        raise ValueError("Only conditions 1, 2, 3, 4, 5 are supported.")

    n = len(sessions_used)
    if n < n_train + n_target + n_test:
        raise ValueError(
            f"Need at least {n_train + n_target + n_test} sessions (after exclusions), got {n}. "
            f"Condition {condition}: {n_train} train + {n_target} val + {n_test} test."
        )
    training_names = sessions_used[:n_train]
    target_names = sessions_used[n_train : n_train + n_target]
    test_names = sessions_used[n_train + n_target : n_train + n_target + n_test]
    return training_names, target_names, test_names, save_suffix


def build_and_save_condition(
    data_dir: str,
    output_dir: str,
    condition: int,
    sessions: list,
    bad_trials_dict: dict = None,
    split_comp_n: int = 80,
    split_seed: int = 42,
) -> str:
    """
    Build train/val/test/competition HDF5 layout and save manifest.
    Writes to output_dir/save_suffix/ so multiple conditions do not overwrite each other.
    - training_days: combined train+val -> .../train/<session>/data.hdf5
    - val (target): combined train+val -> .../val/<session>/data.hdf5
    - test: combined train+val -> .../test/<session>/data.hdf5
    - competition: from data_test.hdf5 -> .../competition/<session>/data.hdf5
    """
    # Only use first 26 sessions (condition_config handles this)
    training_names, target_names, test_names, save_suffix = condition_config(condition, sessions)
    n_used = len(training_names) + len(target_names) + len(test_names)
    print(f"\nUsing first {n_used} sessions: {len(training_names)} train, {len(target_names)} target/val, {len(test_names)} test")
    rng = np.random.RandomState(split_seed + int(condition))
    condition_output_dir = os.path.join(output_dir, save_suffix.lstrip("_"))

    manifest = {
        "train": [],
        "val": [],
        "test": [],
        "competition": [],
        "training_days": training_names,
        "target_days": target_names,
        "val_days": target_names,
        "test_days": test_names,
        "condition": condition,
        "save_suffix": save_suffix,
    }

    # Training sessions: if train+val exist, combine; else just train
    print("Train sessions:")
    train_paths_out = []
    for session in training_names:
        train_path = os.path.join(data_dir, session, "data_train.hdf5")
        val_path = os.path.join(data_dir, session, "data_val.hdf5")
        out_path = os.path.join(condition_output_dir, "train", session, "data.hdf5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        trials_train = get_trial_indices_from_hdf5(train_path, bad_trials_dict, session)
        trials_val = get_trial_indices_from_hdf5(val_path, bad_trials_dict, session)
        
        if trials_val:  # If val exists, combine train+val
            total = combine_and_write_hdf5(
                [train_path, val_path],
                [trials_train, trials_val],
                out_path,
                session_name=session,
            )
            print(f"  train/{session}: {total} trials (train={len(trials_train)}, val={len(trials_val)})")
        else:  # Else just train
            total = combine_and_write_hdf5(
                [train_path],
                [trials_train],
                out_path,
                session_name=session,
            )
            print(f"  train/{session}: {total} trials (train only, no val)")
        
        if total > 0:
            train_paths_out.append(out_path)
    manifest["train"] = train_paths_out

    # Target sessions: val = train+val, competition = test
    print("Val sessions (train+val for target sessions):")
    val_paths_out = []
    for session in target_names:
        train_path = os.path.join(data_dir, session, "data_train.hdf5")
        val_path = os.path.join(data_dir, session, "data_val.hdf5")
        out_path = os.path.join(condition_output_dir, "val", session, "data.hdf5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        trials_train = get_trial_indices_from_hdf5(train_path, bad_trials_dict, session)
        trials_val = get_trial_indices_from_hdf5(val_path, bad_trials_dict, session)
        total = combine_and_write_hdf5(
            [train_path, val_path],
            [trials_train, trials_val],
            out_path,
            session_name=session,
        )
        if total > 0:
            val_paths_out.append(out_path)
        print(f"  val/{session}: {total} trials (train={len(trials_train)}, val={len(trials_val)})")
    manifest["val"] = val_paths_out

    # Competition = target from data_test.hdf5 (same sessions as target)
    print("Competition (target from data_test.hdf5):")
    comp_paths_out = []
    for session in target_names:
        test_path = os.path.join(data_dir, session, "data_test.hdf5")
        out_comp = os.path.join(condition_output_dir, "competition", session, "data.hdf5")
        os.makedirs(os.path.dirname(out_comp), exist_ok=True)
        
        if os.path.exists(test_path):
            # Use data_test.hdf5 directly
            shutil.copy2(test_path, out_comp)
            n_comp = len(get_trial_indices_from_hdf5(out_comp, None, session))
            comp_paths_out.append(out_comp)
            print(f"  competition/{session}: {n_comp} trials (from data_test.hdf5)")
        else:
            print(f"  competition/{session}: WARNING - data_test.hdf5 not found, skipping")
    manifest["competition"] = comp_paths_out

    # Test sessions: use train+val
    print("Test sessions (train+val for test sessions):")
    test_paths_out = []
    for session in test_names:
        train_path = os.path.join(data_dir, session, "data_train.hdf5")
        val_path = os.path.join(data_dir, session, "data_val.hdf5")
        out_path = os.path.join(condition_output_dir, "test", session, "data.hdf5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        trials_train = get_trial_indices_from_hdf5(train_path, bad_trials_dict, session)
        trials_val = get_trial_indices_from_hdf5(val_path, bad_trials_dict, session)
        total = combine_and_write_hdf5(
            [train_path, val_path],
            [trials_train, trials_val],
            out_path,
            session_name=session,
        )
        if total > 0:
            test_paths_out.append(out_path)
        print(f"  test/{session}: {total} trials (train={len(trials_train)}, val={len(trials_val)})")
    manifest["test"] = test_paths_out

    # Build trial_indicies for each split (format expected by BrainToTextDataset)
    def build_trial_indicies(paths: list) -> dict:
        out = {}
        for day_idx, path in enumerate(paths):
            if not os.path.exists(path):
                continue
            n = 0
            with h5py.File(path, "r") as f:
                n = sum(1 for k in f.keys() if k.startswith("trial_"))
            out[day_idx] = {"trials": list(range(n)), "session_path": path}
        return out

    manifest["train_trial_indicies"] = build_trial_indicies(manifest["train"])
    manifest["val_trial_indicies"] = build_trial_indicies(manifest["val"])
    manifest["test_trial_indicies"] = build_trial_indicies(manifest["test"])
    manifest["competition_trial_indicies"] = build_trial_indicies(manifest["competition"])

    # Manifest pickle (same idea as transformer: paths and metadata for loaders)
    manifest_path = os.path.join(condition_output_dir, f"manifest{save_suffix}.pkl")
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f)
    print(f"Saved manifest -> {manifest_path}")
    return manifest_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Format brain-to-text HDF5 into train/val/test + competition (data_test.hdf5 as target)."
    )
    p.add_argument("--data-dir", type=str, required=True, help="Base dir with session/data_train.hdf5, data_val.hdf5, and data_test.hdf5")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for formatted train/val/test/competition")
    p.add_argument(
        "--conditions",
        nargs="+",
        type=int,
        default=[1, 2],
        help=(
            "Condition id(s): 1 = 5/13/8 (26 sess); 2 = 10/8/8 (26 sess); "
            "3 = 5/5/8 (18 sess); 4 = 5/5/3 (13 sess); 5 = 10/5/5 (20 sess). "
            "Default: 1 2"
        ),
    )
    p.add_argument(
        "--sessions",
        nargs="+",
        type=str,
        default=None,
        help="Session names; default uses built-in t15 list.",
    )
    p.add_argument(
        "--sessions-file",
        type=str,
        default=None,
        help="Path to file with one session name per line (overrides --sessions).",
    )
    p.add_argument("--split-comp-n", type=int, default=80, help="Trials to use for competition when no competitionHoldOut file.")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--bad-trials",
        type=str,
        default=None,
        help="Path to pickle of bad_trials_dict {session: {block_num: [trial_nums]}}.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.sessions_file:
        with open(args.sessions_file) as f:
            sessions = [line.strip() for line in f if line.strip()]
    elif args.sessions:
        sessions = args.sessions
    else:
        sessions = DEFAULT_SESSIONS_T15

    if SESSIONS_SKIP_NO_VAL_TEST:
        print(
            f"Including {len(SESSIONS_SKIP_NO_VAL_TEST)} sessions with no data_val.hdf5 (train-only source): "
            f"{sorted(SESSIONS_SKIP_NO_VAL_TEST)}"
        )

    # Only include sessions that exist under data_dir
    existing = []
    for s in sessions:
        train_p = os.path.join(args.data_dir, s, "data_train.hdf5")
        val_p = os.path.join(args.data_dir, s, "data_val.hdf5")
        if os.path.exists(train_p) or os.path.exists(val_p):
            existing.append(s)
    if len(existing) < len(sessions):
        print(f"Using {len(existing)} sessions with existing data (skipped {len(sessions) - len(existing)} missing).")
    sessions = existing

    bad_trials_dict = None
    if args.bad_trials and os.path.exists(args.bad_trials):
        with open(args.bad_trials, "rb") as f:
            bad_trials_dict = pickle.load(f)

    for condition in args.conditions:
        if condition not in (1, 2, 3, 4, 5):
            print(f"Skipping unsupported condition {condition} (only 1, 2, 3, 4, 5 are supported).")
            continue
        build_and_save_condition(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            condition=condition,
            sessions=sessions,
            bad_trials_dict=bad_trials_dict,
            split_comp_n=args.split_comp_n,
            split_seed=args.split_seed,
        )
        print(f"Saved condition {condition}.")


if __name__ == "__main__":
    main()
