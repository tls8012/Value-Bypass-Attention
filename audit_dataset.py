from datasets import load_from_disk
import numpy as np

from configs import DATASET_PATH, SEQ_LONG, SEQ_WARM
from boilerplate_code import build_tokenizer

def audit_basic(ds, tokenizer, sample_size=5000):
    # 샘플링
    n = min(len(ds), sample_size)
    ds_s = ds.shuffle(seed=0).select(range(n))

    pad_id  = tokenizer.pad_token_id
    unk_id  = tokenizer.unk_token_id
    mask_id = tokenizer.mask_token_id
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    V       = tokenizer.vocab_size

    lengths = []
    unk_cnt = 0
    mask_cnt = 0
    pad_cnt = 0
    bad_range_cnt = 0
    cls_ok = 0
    sep_ok = 0

    for ex in ds_s:
        ids = ex["input_ids"]
        am  = ex.get("attention_mask", [1] * len(ids))

        lengths.append(len(ids))
        unk_cnt  += sum(1 for t in ids if t == unk_id) if unk_id is not None else 0
        mask_cnt += sum(1 for t in ids if t == mask_id) if mask_id is not None else 0
        pad_cnt  += sum(1 for t in ids if t == pad_id)  if pad_id  is not None else 0
        bad_range_cnt += sum(1 for t in ids if (t < 0 or t >= V))

        if cls_id is not None and len(ids) > 0 and ids[0] == cls_id:
            cls_ok += 1
        if sep_id is not None and (len(ids) > 1) and (ids[-1] == sep_id):
            sep_ok += 1

        # attention_mask 길이 정합성도 가볍게
        assert len(am) == len(ids), "attention_mask length mismatch"

    L = np.array(lengths)
    total_tokens = L.sum()

    report = {
        "num_samples": n,
        "len_mean": float(L.mean()),
        "len_min": int(L.min()),
        "len_max": int(L.max()),
        "total_tokens": int(total_tokens),
        "unk_ratio": float(unk_cnt / max(total_tokens, 1)),
        "pre_mask_ratio": float(mask_cnt / max(total_tokens, 1)),   # 사전 토큰화 데이터엔 0이어야 정상
        "pad_ratio": float(pad_cnt / max(total_tokens, 1)),         # padding=False였으니 0이 정상
        "bad_id_ratio": float(bad_range_cnt / max(total_tokens, 1)),
        "starts_with_CLS_ratio": float(cls_ok / n) if n else 0.0,
        "ends_with_SEP_ratio": float(sep_ok / n) if n else 0.0,
        "vocab_size": V,
        "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
    }
    return report

def audit_paths(paths, tokenizer, sample_size=5000):
    out = {}
    for name, path in paths.items():
        ds = load_from_disk(path)
        out[name] = audit_basic(ds, tokenizer, sample_size=sample_size)
    return out

# 사용 예시
tok = build_tokenizer()  # 학습에 쓰는 것과 '동일한' 토크나이저
paths = {
  "warm_train": f"{DATASET_PATH}/{SEQ_WARM}/train",
  "warm_val":   f"{DATASET_PATH}/{SEQ_WARM}/val",
  "long_train": f"{DATASET_PATH}/{SEQ_LONG}/train",
  "long_val":   f"{DATASET_PATH}/{SEQ_LONG}/val",
}
print(audit_paths(paths, tok, sample_size=5000))
