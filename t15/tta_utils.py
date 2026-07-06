import torch
import re
import numpy as np
import math
from torch.utils.data import Subset
from g2p_en import G2p
g2p = G2p()  # <- Global instance
import nltk
from nltk.data import find
from torch.nn.utils.rnn import pad_sequence


def convert_sentence(s):
    
    s = s.lower()
    charMarks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 "'", ' ']
    ans = []
    for i in s:
        if(i in charMarks):
            ans.append(i)
    
    return ''.join(ans)

def compute_lambda(memo_loss: torch.Tensor, D: int, gamma: float = 1.0) -> torch.Tensor:
    max_entropy = math.log(D)
    norm_entropy = memo_loss / max_entropy
    lambda_val = (1.0 - norm_entropy).clamp(min=0.0, max=1.0)  # safety clamp
    return lambda_val ** gamma

def clean_transcription(text):
    
    """
    Cleans a transcription string by:
    1. Removing leading/trailing whitespace
    2. Removing all characters except letters, hyphens, spaces, and apostrophes
    3. Removing double hyphens
    4. Converting to lowercase
    """
    
    text = str(text).strip()
    text = re.sub(r"[^a-zA-Z\- ']", '', text)
    text = text.replace('--', '')
    return text.lower()

def get_phonemes(thisTranscription):
    try:
        find("taggers/averaged_perceptron_tagger_eng")
        find("tokenizers/punkt")
    except LookupError:
        # Ensure required NLTK assets for g2p_en on first use
        nltk.download("averaged_perceptron_tagger_eng")
        nltk.download("punkt")
    
    phonemes = []
    
    for p in g2p(thisTranscription):
        
        if p == ' ':
            phonemes.append('SIL')
        p = re.sub(r'[0-9]', '', p)  # Remove stress
        if re.match(r'^[A-Z]+$', p):  # Only keep phonemes (uppercase only)
            phonemes.append(p)
    
    phonemes.append('SIL')  # Add trailing SIL
    
    PHONE_DEF = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'B',  'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G',
        'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V',
        'W', 'Y', 'Z', 'ZH','SIL'
    ]
    
    PHONE_DEF_SIL = PHONE_DEF + ['SIL']

    phoneme_ids = [PHONE_DEF_SIL.index(p) + 1 for p in phonemes]

    return torch.tensor(phoneme_ids, dtype=torch.long), torch.tensor([len(phoneme_ids)], dtype=torch.long)

def get_data_file(path):
    
    suffix_map = {
        "data_log_both": "/data/willett_data/ptDecoder_ctc_both",
        "data": "/data/willett_data/ptDecoder_ctc",
        "data_log_both_held_out_days": "/data/willett_data/ptDecoder_ctc_both_held_out_days",
        "data_log_both_held_out_days_1": "/data/willett_data/ptDecoder_ctc_both_held_out_days_1",
        "data_log_both_held_out_days_2": "/data/willett_data/ptDecoder_ctc_both_held_out_days_2",
    }
    suffix = path.rsplit('/', 1)[-1]
    return suffix_map.get(suffix, path)

def reverse_dataset(dataset):
    return Subset(dataset, list(reversed(range(len(dataset)))))

def _pad_or_stack(items, pad_value=0):
    if torch.is_tensor(items[0]) and items[0].dim() > 0:
        return pad_sequence(items, batch_first=True, padding_value=pad_value)
    if torch.is_tensor(items[0]):
        return torch.stack(items)
    return list(items)

def _padding(batch):
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        collated = {}
        for k in keys:
            values = [x[k] for x in batch]
            if torch.is_tensor(values[0]):
                pad_value = 0
                if k == "seq_class_ids":
                    pad_value = 0
                collated[k] = _pad_or_stack(values, pad_value=pad_value)
            else:
                collated[k] = values
        return collated

    n_fields = len(batch[0])

    if n_fields == 5:
        # (X, y, X_len, y_len, days)
        X, y, X_lens, y_lens, days = zip(*batch)
        transcript = None
    elif n_fields == 6:
        # (X, y, X_len, y_len, days, transcript)
        X, y, X_lens, y_lens, days, transcript = zip(*batch)
    elif n_fields == 7:
        # (X, y, X_len, y_len, days, y2, y2_len) -> drop y2 for TTA
        X, y, X_lens, y_lens, days, y2, y2_lens = zip(*batch)
        y, y_lens = y2, y2_lens
        transcript = None
    elif n_fields == 8:
        # (X, y, X_len, y_len, days, transcript, y2, y2_len) -> drop y2 for TTA
        X, y, X_lens, y_lens, days, transcript, y2, y2_lens = zip(*batch)
        y, y_lens = y2, y2_lens
    else:
        raise ValueError(f"Unexpected batch structure with {n_fields} fields")

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    if transcript is None:
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
        transcript
    )
        
def get_dataloader(dataset, batch_size=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=False, num_workers=0, collate_fn=_padding)

def decode_sequence(pred, adjusted_len):
    pred = torch.argmax(pred[:adjusted_len], dim=-1)
    pred = torch.unique_consecutive(pred)
    return np.array([i for i in pred.cpu().numpy() if i != 0])
