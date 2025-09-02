#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Overfit Trainer for BERT-Style MLM (standalone)
- Tiny subset (default: 4096 sequences) to verify pipeline correctness
- Logs to Weights & Biases: loss, ppl, lr, acc@1, mask_rate, pad_ratio
- Asserts weight tying at runtime
- AdamW with no weight decay on LayerNorm/bias (BERT best practice)
"""

import os, math, time, random, json, argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
import wandb

from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# --- Project-local imports (kept as-is; your project must provide these) ---
try:
    from boilerplate_code import build_tokenizer  # must return a HF tokenizer aligned with the model
except Exception as e:
    raise SystemExit(f"[FATAL] Could not import build_tokenizer from boilerplate_code: {e}")

try:
    from load_model import load_mlm_model  # function that returns a nn.Module for MLM
except Exception as e:
    load_mlm_model = None
    
from boilerplate_code import BertMiniForMLM, BertMiniConfig

try:
    from configs import DATASET_PATH, SEQ_WARM, SEQ_LONG, WANDB_PROJECT
except Exception:
    DATASET_PATH, SEQ_WARM, SEQ_LONG, WANDB_PROJECT = "./data", 128, 512, "mlm-debug"


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_collator(tokenizer, mlm_probability=0.15):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )


def assert_tied_embeddings(model: nn.Module):
    """Ensure decoder weight is tied to input embeddings."""
    try:
        emb_w = model.bert.embeddings.word_embeddings.weight
        dec_w = model.decoder.weight
    except Exception as e:
        raise AssertionError(f"Cannot access embeddings/decoder to verify tie: {e}")

    assert dec_w.data_ptr() == emb_w.data_ptr(), \
        "decoder weight is NOT tied to embeddings (data_ptr mismatch)"
    return True


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01,
                    betas=(0.9, 0.999), eps: float = 1e-8) -> torch.optim.Optimizer:
    """AdamW with no weight decay on LayerNorm/bias (BERT best practice)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "layernorm" in n.lower() or "layer_norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas, eps=eps
    )
    return opt


def cycle(dl):
    while True:
        for b in dl:
            yield b


def pick_dataset_path(seq_len: int, split: str) -> str:
    sub = str(seq_len)
    return os.path.join(DATASET_PATH, sub, split)


def get_subset_loader(tokenizer, seq_len: int, split: str, subset: int, batch_size: int,
                      mlm_p: float, num_workers: int = 4, seed: int = 1234) -> DataLoader:
    ds_path = pick_dataset_path(seq_len, "train" if split == "train" else "val")
    if not os.path.exists(ds_path):
        raise SystemExit(f"[FATAL] tokenized dataset not found: {ds_path}")

    ds = load_from_disk(ds_path)
    n = len(ds)
    subset = min(subset, n) if subset > 0 else n
    ds = ds.shuffle(seed=seed).select(range(subset))

    collator = build_collator(tokenizer, mlm_probability=mlm_p)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collator, num_workers=num_workers, pin_memory=True)
    return dl

@torch.no_grad()
def logits_entropy_probe(model, batch):
    out = model(input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(), labels=None)
    logits = out["logits"]
    mask = batch["labels"].cuda().ne(-100)
    p = torch.log_softmax(logits[mask].float(), dim=-1)
    H = -(p.exp()*p).sum(-1).mean().item()
    std = logits[mask].float().std().item()
    print({"entropy_nats": H, "logit_std": std})

def main():
    parser = argparse.ArgumentParser(description="Quick Overfit Trainer (MLM)")
    parser.add_argument("--seq", type=str, default="warm", choices=["warm", "long"],
                        help="Use tokenized dataset with this sequence length (warm=SEQ_WARM, long=SEQ_LONG).")
    parser.add_argument("--subset", type=int, default=496, help="Number of sequences to train on.")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for non-LN/non-bias.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (of total steps).")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine"],
                        help="LR schedule (no restarts).")
    parser.add_argument("--mlm_p", type=float, default=0.15, help="MLM masking probability.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--project", type=str, default="OVERFIT", help="W&B project.")
    parser.add_argument("--run_name", type=str, default="quick-overfit-smaller", help="W&B run name.")
    parser.add_argument("--outdir", type=str, default="./overfit_ckpts", help="Where to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer()
    
    model = BertMiniForMLM(BertMiniConfig())
    
    #torch.backends.cuda.enable_flash_sdp(False)
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_math_sdp(True)

    def apply_bert_init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)  # ★ BERT std
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    model.apply(apply_bert_init)

    # tie 다시 묶기(가중치 재초기화했으니 안전하게)
    model.decoder.weight = model.bert.embeddings.word_embeddings.weight
    
    
    model.to(device)
    model.train()

    # Sanity: assert tied embeddings
    try:
        assert_tied_embeddings(model)
        tied_ok = True
    except AssertionError as e:
        tied_ok = False
        print(f"[WARN] {e}")

    # Choose seq length group
    seq_len = SEQ_WARM if args.seq == "warm" else SEQ_LONG

    # Dataloader on a tiny subset
    train_loader = get_subset_loader(tokenizer, seq_len, "train",
                                     subset=args.subset, batch_size=args.batch_size,
                                     mlm_p=args.mlm_p, num_workers=4)

    # Optim & sched
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    warmup = max(1, int(args.steps * args.warmup_ratio))
    if args.scheduler == "linear":
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=args.steps)
    else:
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=args.steps)

    os.makedirs(args.outdir, exist_ok=True)

    # W&B
    wandb.init(project=args.project, name=args.run_name, config={
        "seq_len": seq_len, "subset": args.subset, "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "scheduler": args.scheduler, "mlm_p": args.mlm_p,
        "bf16": args.bf16, "seed": args.seed, "tied_embeddings": tied_ok
    })
    wandb.watch(model, log="gradients", log_freq=100, log_graph=False)

    # Training loop
    it = cycle(train_loader)
    if args.bf16:
        if device.type == "cuda":
            scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            scaler_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        # disabled autocast
        class _NullCtx:
            def __enter__(self): pass
            def __exit__(self, *a): pass
        scaler_ctx = _NullCtx()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with scaler_ctx:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            logits = out["logits"] if isinstance(out, dict) else out.logits
            
        """tau = 8.0  # 5~10 사이부터 시도
        logits = out["logits"] if isinstance(out, dict) else out.logits
        logits = logits / tau
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            ignore_index=-100,
        )"""

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        # Metrics
        with torch.no_grad():
            mask = labels.ne(-100)
            if mask.any():
                acc = (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()
                ppl = math.exp(loss.item())
            else:
                acc = float("nan")
                ppl = float("nan")
            pad_id = tokenizer.pad_token_id
            pad_ratio = (input_ids == pad_id).float().mean().item() if pad_id is not None else 0.0
            mask_rate = mask.float().mean().item()

        if step % 10 == 0 or step == 1:
            wandb.log({
                "step": step,
                "train/loss": loss.item(),
                "train/ppl": ppl,
                "train/lr": sched.get_last_lr()[0],
                "train/mlm_acc@1": acc,
                "train/mask_rate": mask_rate,
                "train/pad_ratio": pad_ratio,
                "elapsed_min": (time.time() - t0) / 60.0
            }, step=step)
        if step % 100 == 0 or step == 1:
            logits_entropy_probe(model, batch)

        if step % 250 == 0 or step == args.steps:
            ckpt_path = os.path.join(args.outdir, f"overfit_step{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

    final_path = os.path.join(args.outdir, "final_overfit.pt")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)

    meta = {
        "final_step": args.steps,
        "seq_len": seq_len,
        "subset": args.subset,
        "best_practice": {
            "optimizer": "AdamW (no WD on LN/bias)",
            "betas": [0.9, 0.999],
            "scheduler": args.scheduler,
            "warmup_ratio": args.warmup_ratio
        }
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Finished quick overfit. Final ckpt: {final_path}")


if __name__ == "__main__":
    main()
