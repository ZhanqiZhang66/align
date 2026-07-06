import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle 
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Optional

from torch.utils.data import ConcatDataset

from torch.utils.data import Sampler

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# SIL token ID: 0=blank, 1-39=phones, 40=SIL (if using PHONE_DEF_SIL with 40 phones)
SIL_TOKEN_ID = 40

class SpeechDataset(Dataset):
    
    def __init__(self, data, transform=None, restricted_days=None,
                 ventral_6v_only=False, return_transcript=False):
        """
        If 'text2' exists in the data, include it (and 'textLens2') in the dataset
        and return them in __getitem__.
        """
        self.data = data
        self.transform = transform
        self.return_transcript = return_transcript
        restricted_days = set(restricted_days or [])

        self.n_days = len(data)

        self.neural_feats = []
        self.text_seqs = []
        self.neural_time_bins = []
        self.text_seq_lens = []
        self.days = []
        self.transcriptions = []
        

        # Always check first day to decide if text2 is present
        self.text2_present = "text2" in data[0]

        if self.text2_present:
            self.text2_seqs = []
            self.text2_seq_lens = []

        for day in range(self.n_days):
            if restricted_days and day not in restricted_days:
                continue

            n_trials = len(data[day]["sentenceDat"])
            for trial in range(n_trials):
                feats = data[day]["sentenceDat"][trial]
                self.neural_feats.append(feats[:, :128] if ventral_6v_only else feats)

                self.text_seqs.append(data[day]["text"][trial])
                self.neural_time_bins.append(feats.shape[0])
                self.text_seq_lens.append(data[day]["textLens"][trial])
                self.transcriptions.append(data[day]['transcriptions'][trial])
                self.days.append(day)

                if self.text2_present:
                    self.text2_seqs.append(data[day]["text2"][trial])
                    self.text2_seq_lens.append(data[day]["textLens2"][trial])

        self.n_trials = len(self.days)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)
        if self.transform:
            neural_feats = self.transform(neural_feats)

        items = [
            neural_feats,
            torch.tensor(self.text_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.text_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        ]

        if self.return_transcript:
            items.append(self.transcriptions[idx])

        if self.text2_present:
            items.extend([
                torch.tensor(self.text2_seqs[idx], dtype=torch.int32),
                torch.tensor(self.text2_seq_lens[idx], dtype=torch.int32),
            ])

        return tuple(items)


class SpeechDatasetNoSIL(SpeechDataset):
    """
    Dataset that filters out SIL tokens from text sequences.
    This ensures ground truth labels don't contain SIL, matching models
    where nClasses doesn't include SIL.
    """
    
    def __init__(self, data, transform=None, restricted_days=None,
                 ventral_6v_only=False, return_transcript=False):
        """
        Same as SpeechDataset, but filters SIL tokens (ID 40) from text sequences.
        """
        self.data = data
        self.transform = transform
        self.return_transcript = return_transcript
        restricted_days = set(restricted_days or [])

        self.n_days = len(data)

        self.neural_feats = []
        self.text_seqs = []
        self.neural_time_bins = []
        self.text_seq_lens = []
        self.days = []
        self.transcriptions = []
        

        # Always check first day to decide if text2 is present
        self.text2_present = "text2" in data[0]

        if self.text2_present:
            self.text2_seqs = []
            self.text2_seq_lens = []

        skipped_empty = 0
        for day in range(self.n_days):
            if restricted_days and day not in restricted_days:
                continue

            n_trials = len(data[day]["sentenceDat"])
            for trial in range(n_trials):
                feats = data[day]["sentenceDat"][trial]

                # Filter out SIL tokens from text sequence
                # Use original textLens to only process the actual sequence (not padding)
                text_seq = data[day]["text"][trial]
                original_len = data[day]["textLens"][trial]
                
                # Slice to actual length if it's a numpy array or tensor
                if isinstance(text_seq, np.ndarray):
                    text_seq = text_seq[:original_len]
                elif hasattr(text_seq, '__len__') and len(text_seq) > original_len:
                    text_seq = text_seq[:original_len]
                
                if isinstance(text_seq, (list, np.ndarray)):
                    text_seq_filtered = [token for token in text_seq if token != SIL_TOKEN_ID]
                else:
                    # If it's already a tensor, convert to list, filter, then back
                    text_seq_list = text_seq.tolist() if hasattr(text_seq, 'tolist') else list(text_seq)
                    text_seq_filtered = [token for token in text_seq_list if token != SIL_TOKEN_ID]
                
                # Skip sequences that become empty after filtering SIL (CTC loss can't handle zero-length targets)
                if len(text_seq_filtered) == 0:
                    skipped_empty += 1
                    continue
                
                self.neural_feats.append(feats[:, :128] if ventral_6v_only else feats)
                self.text_seqs.append(text_seq_filtered)
                self.neural_time_bins.append(feats.shape[0])
                # Update length after filtering SIL
                self.text_seq_lens.append(len(text_seq_filtered))
                self.transcriptions.append(data[day]['transcriptions'][trial])
                self.days.append(day)

                if self.text2_present:
                    # Filter out SIL tokens from text2 sequence
                    # Use original textLens2 to only process the actual sequence (not padding)
                    text2_seq = data[day]["text2"][trial]
                    original_len2 = data[day]["textLens2"][trial]
                    
                    # Slice to actual length if it's a numpy array or tensor
                    if isinstance(text2_seq, np.ndarray):
                        text2_seq = text2_seq[:original_len2]
                    elif hasattr(text2_seq, '__len__') and len(text2_seq) > original_len2:
                        text2_seq = text2_seq[:original_len2]
                    
                    if isinstance(text2_seq, (list, np.ndarray)):
                        text2_seq_filtered = [token for token in text2_seq if token != SIL_TOKEN_ID]
                    else:
                        text2_seq_list = text2_seq.tolist() if hasattr(text2_seq, 'tolist') else list(text2_seq)
                        text2_seq_filtered = [token for token in text2_seq_list if token != SIL_TOKEN_ID]
                    
                    # If text2 becomes empty, use a single blank token (0) to avoid zero-length
                    if len(text2_seq_filtered) == 0:
                        text2_seq_filtered = [0]  # Use blank token as placeholder
                    
                    self.text2_seqs.append(text2_seq_filtered)
                    # Update length after filtering SIL
                    self.text2_seq_lens.append(len(text2_seq_filtered))
        
        if skipped_empty > 0:
            print(f"⚠️  Warning: Skipped {skipped_empty} sequences that became empty after filtering SIL tokens")

        self.n_trials = len(self.days)

def time_stretch(feats: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Time‑stretch or ‑squeeze a single trial.

    Args
    ----
    feats : (T, F) tensor
    factor : float
        > 1.0  -> stretch (longer sequence)
        < 1.0  -> squeeze (shorter sequence)
    """
    T, Fdim = feats.shape
    if T <= 1 or factor == 1.0:
        return feats

    # Use linear interpolation along the time dimension.
    x = feats.transpose(0, 1).unsqueeze(0)  # (1, F, T)
    new_T = max(1, int(round(T * factor)))
    x_stretched = F.interpolate(x, size=new_T, mode="linear", align_corners=False)
    return x_stretched.squeeze(0).transpose(0, 1)  # (new_T, F)


def _canonicalize_stretch_range(stretch_range, default: float = 2.0):
    """
    Convert a stretch range specification into a (min_factor, max_factor) tuple.

    Accepts:
      - scalar (int/float): treated as [scalar, scalar]
      - 2‑element list/tuple: [min_factor, max_factor]
    """
    if stretch_range is None:
        return float(default), float(default)
    if isinstance(stretch_range, (int, float)):
        val = float(stretch_range)
        return val, val
    if isinstance(stretch_range, (list, tuple)) and len(stretch_range) == 2:
        f_min, f_max = float(stretch_range[0]), float(stretch_range[1])
        return f_min, f_max
    raise ValueError("stretch_range must be a scalar or a 2‑element list/tuple.")


def generate_prolonged_samples(
    base_ds: SpeechDataset,
    combined_range=(1, 5),
    stretch_range=2.0,
    sample_size: Optional[int] = None,
):
    """
    Generate new prolonged samples by randomly combining K DIFFERENT samples
    (and optionally stretching them) from a dataset.

    For every item in `base_ds` we construct **one** new prolonged sample
    whose feature sequence is the concatenation of K segments, where
    K is drawn uniformly from `combined_range` (inclusive). Each segment
    comes from a *different* dataset index (up to the dataset size) and is
    either:
      - the original trial, or
      - a time‑stretched version of that trial (using `time_stretch`)

    The label sequences (`text`, and `text2` if present) are concatenated
    across the K chosen samples so that the targets remain aligned with
    the prolonged feature sequence. The returned list therefore has the
    **same length** as `base_ds`.

    Args
    ----
    base_ds : SpeechDataset
        The base dataset to augment. It should be an instance of
        `SpeechDataset` or a compatible wrapper whose items follow the
        `SpeechDataset.__getitem__` layout.
    combined_range : Tuple[int, int]
        (min_K, max_K). A new prolonged sample is formed by concatenating
        K in [min_K, max_K] DIFFERENT original / stretched samples.
    stretch_range : float or (float, float)
        Controls the time‑stretch factor used when creating stretched
        segments. If a scalar (e.g. 2.0), all stretched segments use that
        factor (i.e. [2.0, 2.0]). If a 2‑element sequence [a, b], each
        stretched segment draws a random factor uniformly from [a, b].

    Returns
    -------
    List[Tuple]
        A list of sample tuples with the same structure as
        `SpeechDataset.__getitem__`, suitable for use with the existing
        `_padding` collate function.
    """
    if not isinstance(combined_range, (list, tuple)) or len(combined_range) != 2:
        raise ValueError("combined_range must be a 2‑element (min, max) sequence.")

    min_k, max_k = combined_range
    min_k = int(min_k)
    max_k = int(max_k)
    if min_k < 1 or max_k < min_k:
        raise ValueError("combined_range must satisfy 1 <= min_k <= max_k.")

    # Normalize stretch_range
    f_min, f_max = _canonicalize_stretch_range(stretch_range, default=2.0)

    # If sample_size is not provided, use the entire dataset.
    if sample_size is None:
        n_total = len(base_ds)
    else:
        # If sample_size is provided, use it as the total number of samples to generate.
        n_total = sample_size

    prolonged_samples = []

    # Global flags describing optional fields for all samples
    has_transcript = getattr(base_ds, "return_transcript", False)
    has_text2 = getattr(base_ds, "text2_present", False)

    for _ in range(n_total):
        # How many segments to concatenate for this new prolonged sample?
        K = torch.randint(min_k, max_k + 1, (1,)).item()
        K = min(K, n_total)  # cannot sample more unique indices than dataset size

        # Choose K DIFFERENT indices
        chosen_indices = random.sample(range(n_total), K)

        # Initialize lists to store the segments for this new prolonged sample
        feat_segments = []
        text_segments = []
        total_y_len = 0
        text2_segments = []
        total_y2_len = 0
        day = None
        transcript = None

        # Iterate over the chosen indices to construct the new prolonged sample
        for j in chosen_indices:
            items = list(base_ds[j])

            neural_feats = items[0]           # (T, F)
            text_seq = items[1]               # (L,)
            X_len = items[2]
            y_len = items[3]
            day_j = items[4]

            # Offset to get the transcription and text2 (if present)
            offset = 5
            transcript_j = None
            if has_transcript:
                transcript_j = items[offset]
                offset += 1
            # Initialize text2 variables
            text2 = text2_len = None
            if has_text2:
                text2 = items[offset]
                text2_len = items[offset + 1]

            # Decide whether to stretch this particular sample
            use_stretch = bool(torch.randint(0, 2, (1,)).item())
            if use_stretch:
                # Sample a random stretch factor from the configured range
                factor = float(torch.empty(1).uniform_(f_min, f_max).item())
                seg_feats = time_stretch(neural_feats, factor=factor)
            else:
                seg_feats = neural_feats.clone()

            feat_segments.append(seg_feats)
            text_segments.append(text_seq)
            total_y_len += int(y_len.item())

            if has_text2 and text2 is not None:
                text2_segments.append(text2)
                total_y2_len += int(text2_len.item())

            # For metadata fields that must remain scalar (day, transcript),
            # just take them from the first chosen sample.
            if day is None:
                day = day_j
            if has_transcript and transcript is None:
                transcript = transcript_j

        # Concatenate along time and label dimensions
        new_neural_feats = torch.cat(feat_segments, dim=0)  # (T_total, F)
        new_X_len = torch.tensor(new_neural_feats.shape[0], dtype=X_len.dtype)

        new_text_seq = torch.cat(text_segments, dim=0)
        new_y_len = torch.tensor(total_y_len, dtype=y_len.dtype)

        new_items = [
            new_neural_feats,
            new_text_seq,
            new_X_len,
            new_y_len,
            day,
        ]

        if has_transcript:
            new_items.append(transcript)

        if has_text2 and text2_segments:
            new_text2 = torch.cat(text2_segments, dim=0)
            new_text2_len = torch.tensor(total_y2_len, dtype=text2_len.dtype)
            new_items.extend([new_text2, new_text2_len])

        prolonged_samples.append(tuple(new_items))

    return prolonged_samples


class StretchSqueezeDataset(Dataset):
    """
    Wraps a SpeechDataset and, for each base index i, exposes three
    logical samples:

      i*3 + 0 : original trial
      i*3 + 1 : time‑stretched by factor `stretch_factor`
      i*3 + 2 : (reserved; could be time‑squeezed variant if enabled)
    
    This keeps the original SpeechDataset unchanged and makes it easy
    to turn the augmentation on/off at the loader level.
    """
    
    def __init__(self, base_ds: SpeechDataset, stretch_range=2.0):
        self.base_ds = base_ds
        self.stretch_min, self.stretch_max = _canonicalize_stretch_range(
            stretch_range, default=2.0
        )

    def __len__(self):
        return 3 * len(self.base_ds)

    def __getitem__(self, idx):
        base_idx = idx // 3
        aug_type = idx % 3

        items = list(self.base_ds[base_idx])

        # items[0] is neural_feats, items[2] is X_len in current SpeechDataset
        neural_feats = items[0]          # (T, F)
        X_len = items[2]                 # scalar tensor

        if aug_type == 1:
            # Stretch by a factor sampled from [stretch_min, stretch_max]
            factor = float(torch.empty(1).uniform_(self.stretch_min, self.stretch_max).item())
            neural_feats = time_stretch(neural_feats, factor=factor)
        # elif aug_type == 2:
        #     neural_feats = time_stretch(neural_feats, factor=0.5)

        # Update the recorded length to match the new time dimension
        X_len = torch.tensor(neural_feats.shape[0], dtype=X_len.dtype)

        items[0] = neural_feats
        items[2] = X_len
        return tuple(items)

class ProlongedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def pad_to_multiple(tensor, multiple, dim=1, value=0):
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)


from torch.utils.data import Sampler
import random


class BalancedSourceBatchSampler:
    """
    BatchSampler that yields batches with balanced samples per source domain.
    Each batch has ~batch_size // n_domains samples from each domain.
    Uses oversampling for minority domains when needed.
    Use with DataLoader(batch_sampler=...).
    """
    def __init__(self, dataset, batch_size, n_domains, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_domains = n_domains
        self.generator = generator

        # Group indices by domain (day)
        self.domain_to_indices = [[] for _ in range(n_domains)]
        for idx in range(len(dataset)):
            day = dataset.days[idx]
            if day < n_domains:
                self.domain_to_indices[day].append(idx)

        self.samples_per_domain = max(1, batch_size // n_domains)
        n_per_batch = self.samples_per_domain * n_domains
        self.total_samples = sum(len(v) for v in self.domain_to_indices)
        self.num_batches = max(1, (self.total_samples + n_per_batch - 1) // n_per_batch)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for d in range(self.n_domains):
                available = self.domain_to_indices[d]
                if len(available) == 0:
                    continue
                n_need = self.samples_per_domain
                if self.generator is not None:
                    chosen = torch.randint(0, len(available), (n_need,), generator=self.generator)
                else:
                    chosen = np.random.randint(0, len(available), size=n_need)
                for i in chosen:
                    batch.append(available[int(i)])
            if batch:
                batch = list(batch)
                random.shuffle(batch)
                yield batch

    def __len__(self):
        return self.num_batches


class ShuffleByBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = len(dataset)

    def __iter__(self):
        n = self.dataset_len
        indices = list(range(n))

        # Step 1: Group into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, n, self.batch_size)]

        # Step 2: Shuffle the batch order
        random.shuffle(batches)

        # Step 3: Yield batches (lists of indices)
        for batch in batches:
            yield batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

def getDatasetLoaders(
    datasetName,
    batchSize,
    restricted_days=[],
    ventral_6v_only=False,
    include_original: bool = True,
    include_stretched_samples: bool = False,
    include_prolonged_samples: bool = False,
    stretch_range=2.0,
    distributed: bool = False,
):

    '''
    Possible types of dataset combo:
    1. original + stretched_samples + prolonged_samples
    2. original + stretched_samples
    3. original + prolonged_samples
    4. stretched_samples + prolonged_samples
    5. original
    6. stretched_samples
    7. prolonged_samples
    '''
    
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        
        if len(batch[0]) == 5:
            # (X, y, X_len, y_len, days)
            X, y, X_lens, y_lens, days = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
            )
        elif len(batch[0]) == 7:
            # (X, y, X_len, y_len, days, y2, y2_len)
            X, y, X_lens, y_lens, days, y2, y2_lens = zip(*batch)
            X_padded  = pad_sequence(X,  batch_first=True, padding_value=0)
            y_padded  = pad_sequence(y,  batch_first=True, padding_value=0)
            y2_padded = pad_sequence(y2, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                y2_padded,
                torch.stack(y2_lens),
            )


    # ===== Build train dataset =====
    base_train_ds = SpeechDataset(
        loadedData["train"],
        transform=None,
        restricted_days=restricted_days,
        ventral_6v_only=ventral_6v_only,
    )
    train_components = []

    # Optional time stretch/squeeze augmentation
    if include_stretched_samples:
        stretch_ds = StretchSqueezeDataset(
            base_train_ds,
            stretch_range=stretch_range,
        )
        train_components.append(stretch_ds)

    # Optional prolonged samples augmentation
    if include_prolonged_samples:
        prolonged_list = generate_prolonged_samples(
            base_train_ds,
            combined_range=(1, 10),
            stretch_range=stretch_range,
        )
        prolonged_ds = ProlongedDataset(prolonged_list)
        train_components.append(prolonged_ds)

    # If no augmentation is applied, just use the original dataset.
    if include_original or (not include_stretched_samples and not include_prolonged_samples):
        train_components.append(base_train_ds)

    # Concatenate train_components
    if len(train_components) > 1:
        train_ds = ConcatDataset(train_components)
    else:
        train_ds = train_components[0]

    # If running under DDP, use a DistributedSampler so each rank sees a
    # different shard of the training data.
    use_distributed = (
        distributed
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    if use_distributed:
        train_sampler = DistributedSampler(train_ds)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    
    # ===== Build test dataset =====
    test_ds = SpeechDataset(
        loadedData["test"],
        restricted_days=restricted_days,
        ventral_6v_only=ventral_6v_only,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
        
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def getDatasetLoadersInterleaved(
    datasetName,
    batchSize, 
    restricted_days=[],
    ventral_6v_only=False,
    interleave_step=10,
):
    """
    Create train and eval datasets by interleaving samples from the training set.
    
    Args:
        datasetName: Path to dataset pickle file
        batchSize: Batch size for data loaders
        restricted_days: List of days to restrict to
        ventral_6v_only: Whether to use only ventral 6v channels
        interleave_step: Every Nth sample goes to eval (default: 10, meaning 10% eval)
    
    Returns:
        tuple: (train_loader, eval_loader, loadedData)
    """
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        
        if len(batch[0]) == 5:
            # (X, y, X_len, y_len, days)
            X, y, X_lens, y_lens, days = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
            )
        elif len(batch[0]) == 7:
            # (X, y, X_len, y_len, days, y2, y2_len)
            X, y, X_lens, y_lens, days, y2, y2_lens = zip(*batch)
            X_padded  = pad_sequence(X,  batch_first=True, padding_value=0)
            y_padded  = pad_sequence(y,  batch_first=True, padding_value=0)
            y2_padded = pad_sequence(y2, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                y2_padded,
                torch.stack(y2_lens),
            )
    
    # Flatten all trials from all days into a list
    all_trials = []
    train_data = loadedData["train"]
    restricted_days_set = set(restricted_days) if restricted_days else set()
    
    # Check if text2 is present
    text2_present = "text2" in train_data[0] if len(train_data) > 0 else False
    
    for day in range(len(train_data)):
        if restricted_days_set and day not in restricted_days_set:
            continue
        
        n_trials = len(train_data[day]["sentenceDat"])
        for trial in range(n_trials):
            trial_data = {
                "sentenceDat": train_data[day]["sentenceDat"][trial],
                "text": train_data[day]["text"][trial],
                "textLens": train_data[day]["textLens"][trial],
                "transcriptions": train_data[day]["transcriptions"][trial],
                "day": day,
            }
            if text2_present:
                trial_data["text2"] = train_data[day]["text2"][trial]
                trial_data["textLens2"] = train_data[day]["textLens2"][trial]
            
            all_trials.append(trial_data)
    
    # Interleave: every Nth sample goes to eval
    train_trials = []
    eval_trials = []
    
    for idx, trial in enumerate(all_trials):
        if idx % interleave_step == 0:
            eval_trials.append(trial)
        else:
            train_trials.append(trial)
    
    # Reconstruct data structure for train and eval sets
    def reconstruct_data_structure(trials):
        """Reconstruct the data structure from a list of trials."""
        # Group trials by day
        days_dict = {}
        for trial in trials:
            day = trial["day"]
            if day not in days_dict:
                days_dict[day] = {
                    "sentenceDat": [],
                    "text": [],
                    "textLens": [],
                    "transcriptions": [],
                }
                if text2_present:
                    days_dict[day]["text2"] = []
                    days_dict[day]["textLens2"] = []
            
            days_dict[day]["sentenceDat"].append(trial["sentenceDat"])
            days_dict[day]["text"].append(trial["text"])
            days_dict[day]["textLens"].append(trial["textLens"])
            days_dict[day]["transcriptions"].append(trial["transcriptions"])
            if text2_present:
                days_dict[day]["text2"].append(trial["text2"])
                days_dict[day]["textLens2"].append(trial["textLens2"])
        
        # Convert to list format matching original structure
        # Create dense list with all days (including empty ones) to preserve day indices
        max_day = max(days_dict.keys()) if days_dict else -1
        data_list = []
        for day in range(max_day + 1):
            if day in days_dict:
                data_list.append(days_dict[day])
            else:
                # Create empty day entry to preserve day index alignment
                empty_day = {
                    "sentenceDat": [],
                    "text": [],
                    "textLens": [],
                    "transcriptions": [],
                }
                if text2_present:
                    empty_day["text2"] = []
                    empty_day["textLens2"] = []
                data_list.append(empty_day)
        
        return data_list
    
    train_data_interleaved = reconstruct_data_structure(train_trials)
    eval_data_interleaved = reconstruct_data_structure(eval_trials)
    
    # Create datasets
    train_ds = SpeechDataset(train_data_interleaved, transform=None, 
                             restricted_days=restricted_days, 
                             ventral_6v_only=ventral_6v_only)
    
    eval_ds = SpeechDataset(eval_data_interleaved, 
                            restricted_days=restricted_days, 
                            ventral_6v_only=ventral_6v_only)
    
    # Check for empty datasets before creating DataLoaders
    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )
    if len(eval_ds) == 0:
        raise ValueError(
            f"Eval dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
        
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    
    # Create a modified loadedData dict for compatibility
    loadedData_interleaved = {
        "train": train_data_interleaved,
        "test": eval_data_interleaved,  # Use eval as "test" for compatibility
    }
    
    print(f"✅ Created interleaved dataset:")
    print(f"   Train samples: {len(train_trials)} ({100 * len(train_trials) / len(all_trials):.1f}%)")
    print(f"   Eval samples: {len(eval_trials)} ({100 * len(eval_trials) / len(all_trials):.1f}%)")
    print(f"   Interleave step: {interleave_step} (every {interleave_step}th sample → eval)")

    return train_loader, eval_loader, loadedData_interleaved

def getDatasetLoadersNoSIL(
    datasetName,
    batchSize, 
    restricted_days=[],
    ventral_6v_only=False,
):
    """
    Dataset loader that uses SpeechDatasetNoSIL to filter out SIL tokens from ground truth.
    This matches models where nClasses doesn't include SIL.
    """
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        
        if len(batch[0]) == 5:
            # (X, y, X_len, y_len, days)
            X, y, X_lens, y_lens, days = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
            )
        elif len(batch[0]) == 7:
            # (X, y, X_len, y_len, days, y2, y2_len)
            X, y, X_lens, y_lens, days, y2, y2_lens = zip(*batch)
            X_padded  = pad_sequence(X,  batch_first=True, padding_value=0)
            y_padded  = pad_sequence(y,  batch_first=True, padding_value=0)
            y2_padded = pad_sequence(y2, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                y2_padded,
                torch.stack(y2_lens),
            )
  
    train_ds = SpeechDatasetNoSIL(loadedData["train"], transform=None, 
                             restricted_days=restricted_days, 
                             ventral_6v_only=ventral_6v_only)
    
    test_ds = SpeechDatasetNoSIL(loadedData["test"], 
                            restricted_days=restricted_days, 
                            ventral_6v_only=ventral_6v_only)

    # Check for empty datasets before creating DataLoaders
    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )
    if len(test_ds) == 0:
        raise ValueError(
            f"Test dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
        
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def getDatasetLoaders_dpo(
    datasetName,
    batchSize, 
    restricted_days=None,
    ventral_6v_only=False,
):
    if restricted_days is None:
        restricted_days = []

    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        """
        Handles:
          5-tuple: (X, y, X_len, y_len, day)
          6-tuple: (X, y, X_len, y_len, day, transcript)
          7-tuple: (X, y, X_len, y_len, day, y2, y2_len)
          8-tuple: (X, y, X_len, y_len, day, transcript, y2, y2_len)
        Transcripts are left as a list of strings (no padding).
        """
        first = batch[0]
        n_fields = len(first)

        if n_fields == 5:
            # (X, y, X_len, y_len, days)
            X, y, X_lens, y_lens, days = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
            )

        elif n_fields == 6:
            # (X, y, X_len, y_len, days, transcript)
            X, y, X_lens, y_lens, days, transcripts = zip(*batch)
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
            y_padded = pad_sequence(y, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                list(transcripts),   # keep as list[str]
            )

        elif n_fields == 7:
            # (X, y, X_len, y_len, days, y2, y2_len)
            X, y, X_lens, y_lens, days, y2, y2_lens = zip(*batch)
            X_padded  = pad_sequence(X,  batch_first=True, padding_value=0)
            y_padded  = pad_sequence(y,  batch_first=True, padding_value=0)
            y2_padded = pad_sequence(y2, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                y2_padded,
                torch.stack(y2_lens),
            )

        elif n_fields == 8:
            # (X, y, X_len, y_len, days, transcript, y2, y2_len)
            X, y, X_lens, y_lens, days, transcripts, y2, y2_lens = zip(*batch)
            X_padded  = pad_sequence(X,  batch_first=True, padding_value=0)
            y_padded  = pad_sequence(y,  batch_first=True, padding_value=0)
            y2_padded = pad_sequence(y2, batch_first=True, padding_value=0)
            return (
                X_padded,
                y_padded,
                torch.stack(X_lens),
                torch.stack(y_lens),
                torch.stack(days),
                list(transcripts),    # keep as list[str]
                y2_padded,
                torch.stack(y2_lens),
            )

        else:
            raise ValueError(f"Unexpected batch element length: {n_fields}")

    # DPO needs transcripts → return_transcript=True
    train_ds = SpeechDataset(
        loadedData["train"],
        transform=None, 
        restricted_days=restricted_days, 
        ventral_6v_only=ventral_6v_only,
        return_transcript=True,
    )
    
    test_ds = SpeechDataset(
        loadedData["test"], 
        transform=None,
        restricted_days=restricted_days, 
        ventral_6v_only=ventral_6v_only,
        return_transcript=True,
    )

    # Check for empty datasets before creating DataLoaders
    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )
    if len(test_ds) == 0:
        raise ValueError(
            f"Test dataset is empty after filtering with restricted_days={restricted_days}. "
            f"Please check that the specified days exist in the dataset."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
        
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def segment_data(data: torch.Tensor, N: int, X_len: torch.Tensor, day_idx: torch.Tensor):
    
    """
    Segments data into time-aligned batches of shape (B', N, F), where each segment
    includes only trials with sufficient valid data (according to X_len). If a trial's
    valid length is between start and end, include the last N-length chunk ending at X_len.

    Args:
        data (torch.Tensor): Input tensor of shape (B, T, F)
        N (int): Length of each time segment
        X_len (torch.Tensor): Valid lengths per trial (B,)
        day_idx (torch.Tensor): Day that each trial from the batch comes from (B, )

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Segments of shape (B', N, F)
            - Corresponding day indices of shape (B',)
    """
    B, T, F = data.shape
    max_len = X_len.max().item()

    for start in range(0, max_len - N + 1, N):
        
        segments = []
        segment_days = []
        end = start + N
        
        for b in range(B):
            
            # get 
            x_len = X_len[b].item()
            
            # no padding issues here because X_len is longer than end.
            if x_len >= end:
                segment = data[b, start:end, :]
                segments.append(segment)
                segment_days.append(day_idx[b])
                
            # if there is still some new signal, but not long enough for a chunk
            # take the last N non padded timesteps.
            elif x_len > start:
                segment = data[b, x_len-N:x_len, :]
                segments.append(segment)
                segment_days.append(day_idx[b])
                
            # if signal has finished, randomly select a chunk to preserve batch size. 
            else:
                max_start = x_len - N
                rand_start = torch.randint(0, max_start + 1, (1,)).item()
                segment = data[b, rand_start:rand_start + N, :]
                segments.append(segment)
                segment_days.append(day_idx[b])

        
        yield torch.stack(segments), torch.stack(segment_days)
        
        
def sliding_chunks(x, chunk_size=32, stride=4):
    """
    x: Tensor of shape (B, T, C)
    Returns: Tensor of shape (B, M, chunk_size, C)
    """
    B, T, C = x.shape

    # Unfold the time dimension (dim=1) using torch.nn.functional.unfold logic
    x = x.unfold(dimension=1, size=chunk_size, step=stride).permute(0, 1, 3, 2)  # (B, M, chunk_size, C)
    return x

def training_batch_generator(trainLoader, args):
    
    if args['batchStyle']:
        
        for i in range(args["nBatch"]):
            
            X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
            
            if i % 100 == 0:
                compute_val = True
            else:
                compute_val = False
                
            yield (
                X.to(args["device"]),
                y.to(args["device"]),
                X_len.to(args["device"]),
                y_len.to(args["device"]),
                dayIdx.to(args["device"]),
                compute_val
            )
            
    else:
        num_batches = len(trainLoader)
        for epoch in range(args["n_epochs"]):
            for batch_idx, (X, y, X_len, y_len, dayIdx) in enumerate(tqdm(trainLoader, desc=f"Training Epoch {epoch}")):
                compute_val = (batch_idx == num_batches - 1)
                yield (
                    X.to(args["device"]),
                    y.to(args["device"]),
                    X_len.to(args["device"]),
                    y_len.to(args["device"]),
                    dayIdx.to(args["device"]),
                    compute_val
                )