import os, requests,datetime

from datasets import load_dataset, load_from_disk

from configs import SEQ_LONG, SEQ_WARM, DATASET_PATH, notify, DATASET_NAME, DATASET_SETTING

from boilerplate_code import build_tokenizer

def map_dataset(
    tokenizer = None,
    val_split: float = 0.02,
    num_proc: int | None=None,   # cached_map일 때만 사용
    map_batch_size: int = 1000,    # cached_map일 때만 사용
):
    tokenizer = build_tokenizer()
    print("tokenizing...")
    paths = [
        os.path.join(DATASET_PATH, str(SEQ_WARM), "train"),
        os.path.join(DATASET_PATH, str(SEQ_WARM), "val"),
        os.path.join(DATASET_PATH, str(SEQ_LONG), "train"),
        os.path.join(DATASET_PATH, str(SEQ_LONG), "val")
    ]
    
    all_ok = True
    for path in paths:
        if not os.path.exists(path):
            all_ok = False
            break
        try:
            _ = load_from_disk(path)
        except Exception:
            all_ok = False
            break
    if all_ok:
        print("already there")
        return
    
    def tok_fn_warm(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQ_WARM,
            padding=False,
            return_overflowing_tokens=True
        )
        
    def tok_fn_long(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQ_LONG,
            padding=False,
            return_overflowing_tokens=True
        )
    
    # 1) 위키 로드 & 분할
    notify(f"Making Dataset Start {datetime.datetime.now()}")
    #  "wikitext", "wikitext-2-raw-v1", split="train[:400]" "wikipedia", "20220301.en"
    ds = load_dataset(DATASET_NAME,DATASET_SETTING, split="train", trust_remote_code=True)
    ds = ds.train_test_split(test_size=0.5, seed=42)
    ds_warm, ds_long = ds["train"], ds["test"]
    
    ds_warm = ds_warm.train_test_split(test_size=val_split, seed=42)
    ds_warm_train, ds_warm_val = ds_warm["train"], ds_warm["test"]
    
    ds_long = ds_long.train_test_split(test_size=val_split, seed=42)
    ds_long_train, ds_long_val = ds_long["train"], ds_long["test"]

    if num_proc is None:
        cpu = os.cpu_count() or 2
        num_proc = max(1, min(4, cpu // 2))  # 코랩 T4면 보통 2
    
    rm_cols_tr = ds_warm_train.column_names
    rm_cols_va = ds_warm_val.column_names

    notify(f"Loading Complete, Mapping {datetime.datetime.now()}")

    ds_warm_train = ds_warm_train.map(
        tok_fn_warm, batched=True, batch_size=map_batch_size,
        num_proc=num_proc, remove_columns=rm_cols_tr
    )
    notify(f"Mapping Warm Train Done {datetime.datetime.now()}")
    ds_warm_val = ds_warm_val.map(
        tok_fn_warm, batched=True, batch_size=map_batch_size,
        num_proc=num_proc, remove_columns=rm_cols_va
    )
    notify(f"Mapping Warm Val Done {datetime.datetime.now()}")
    
    rm_cols_tr = ds_long_train.column_names
    rm_cols_va = ds_long_val.column_names

    ds_long_train = ds_long_train.map(
        tok_fn_long, batched=True, batch_size=map_batch_size,
        num_proc=num_proc, remove_columns=rm_cols_tr
    )
    notify(f"Mapping Long Train Done {datetime.datetime.now()}")
    ds_long_val = ds_long_val.map(
        tok_fn_long, batched=True, batch_size=map_batch_size,
        num_proc=num_proc, remove_columns=rm_cols_va
    )
    notify(f"Mapping Long Val Done {datetime.datetime.now()}")
    # 6) 디스크 저장 (여러 프로세스에서 함께 읽게 될 경로)
    out_warm_train = os.path.join(DATASET_PATH, str(SEQ_WARM), "train")
    out_warm_val   = os.path.join(DATASET_PATH, str(SEQ_WARM), "val")
    out_long_train = os.path.join(DATASET_PATH, str(SEQ_LONG), "train")
    out_long_val   = os.path.join(DATASET_PATH, str(SEQ_LONG), "val")

    os.makedirs(out_warm_train, exist_ok=True)
    os.makedirs(out_warm_val,   exist_ok=True)
    os.makedirs(out_long_train, exist_ok=True)
    os.makedirs(out_long_val,   exist_ok=True)

    ds_warm_train.save_to_disk(out_warm_train)
    ds_warm_val.save_to_disk(out_warm_val)
    ds_long_train.save_to_disk(out_long_train)
    ds_long_val.save_to_disk(out_long_val)

if __name__=="__main__":
    map_dataset()