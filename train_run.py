import os, math, time, json, random
from dataclasses import dataclass
from typing import Dict, Any

import requests, datetime

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader

from boilerplate_code import *

from builder import make_model

import wandb_code

from configs import DEVICE
from configs import (
    SEQ_WARM, SEQ_LONG, WARM_STEPS, LONG_STEPS, 
    BATCH, LR, SAVE_EVERY, LOG_EVERY, WANDB_PROJECT, GRAD_ACCUM_STEPS, notify
)

# =========================================================
# 4) 학습 루프 (싱글 GPU면 그냥 파이썬 for 루프 OK)
# =========================================================
@dataclass
class TrainCfg:
    seq_len_warm: int = SEQ_WARM
    seq_len_long: int = SEQ_LONG
    warm_steps: int = WARM_STEPS         # 총 스텝 중 앞부분 L=128
    long_steps: int = LONG_STEPS         # 뒷부분 L=512
    batch_size: int = BATCH              # 시퀀스 배치(메모리에 맞게 조절)
    lr: float = LR
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_every: int = SAVE_EVERY
    grad_accum_steps: int = GRAD_ACCUM_STEPS
    outdir: str = "./ckpts_mini_baseline"

def train(model_in_training: nn.Module, tok, cfg: TrainCfg):
    model_in_training.to(DEVICE)
    model_in_training.train()

    total_steps = cfg.warm_steps + cfg.long_steps
    decay, no_decay = [], []
    for n, p in model_in_training.named_parameters():
        if not p.requires_grad: 
            continue
        if p.ndim == 1 or n.endswith(".bias") or "layernorm" in n.lower() or "layer_norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [ # remove decay for ln? idk
            {"params": decay,    "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    #opt = torch.optim.AdamW(model_in_training.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)

    # LR 스케줄러
    num_warmup = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup, total_steps)
    # cosine kick cos why not
    """num_cycles = 5  # or 3

    sched = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
        num_cycles=num_cycles
    )"""

    collator = build_collator(tok)

    def make_loader(seq_len, shard="train"):
        ds_train, ds_val = load_corpus(tok, seq_len=seq_len, val_split=0.02)
        ds = ds_train if shard == "train" else ds_val
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(shard == "train"),
            num_workers=4, pin_memory=True, collate_fn=collator
        )

    # 두 단계: L=128 → L=512
    loaders = [
        ("warm", cfg.warm_steps, make_loader(cfg.seq_len_warm, "train"), make_loader(cfg.seq_len_warm, "val")),
        ("long", cfg.long_steps, make_loader(cfg.seq_len_long, "train"), make_loader(cfg.seq_len_long, "val")),
    ]

    os.makedirs(cfg.outdir, exist_ok=True)
    scaler = None  # bf16은 GradScaler 불필요; fp16이면 torch.cuda.amp.GradScaler() 사용

    step_global = 0
    ce_best = float("inf")
    t0 = time.time()
    t_last = time.time()

    for phase_name, phase_steps, loader_tr, loader_val in loaders:
        it = iter(loader_tr)
        for _ in range(phase_steps):
            step_global += 1
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader_tr)
                batch = next(it)

            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model_in_training(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"]
            """# ---- k번 마이크로스텝 누적 ----
            for k in range(cfg.grad_accum_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader_tr)
                    batch = next(it)

                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model_in_training(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out["loss"] / cfg.grad_accum_steps     # 누적 위해 분할

                loss.backward()"""

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_in_training.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step_global % LOG_EVERY == 0:
                elapsed = time.time() - t0
                msg = f"[{phase_name}] step {step_global}/{total_steps}  loss {loss.item():.4f}  elapsed {elapsed/60:.1f}m"
                if step_global % (10*LOG_EVERY) == 0:
                    notify(msg)
                L = cfg.seq_len_warm if phase_name == "warm" else cfg.seq_len_long
                batch_tokens = cfg.batch_size * L
                now = time.time()
                step_sec = now - t_last
                t_last = now
                wandb_code.wandb_log_train(step_global, loss.item(), sched.get_last_lr()[0], batch_tokens, step_sec)
                
                with torch.no_grad():
                    logits = out["logits"].detach()
                    mask = labels != -100
                    acc = (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()
                    wandb_code.wandb_log_general({"mlm_acc@1": round(acc, 4)}, step=step_global)
                    
                    mask_rate = (labels != -100).float().mean().item()
                    pad_ratio = (input_ids == tok.pad_token_id).float().mean().item() if tok.pad_token_id is not None else 0.0
                    wandb_code.wandb_log_general({"mask_rate": round(mask_rate, 4), "pad_ratio": round(pad_ratio, 4)}, step=step_global)
                    
            if step_global % cfg.save_every == 0:
                # 검증(간단히 CE만)
                model_in_training.eval()
                total, n = 0.0, 0
                with torch.no_grad():
                    for vb in loader_val:
                        vi = vb["input_ids"].to(DEVICE)
                        vm = vb["attention_mask"].to(DEVICE)
                        vl = vb["labels"].to(DEVICE)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            vo = model_in_training(input_ids=vi, attention_mask=vm, labels=vl)
                            total += vo["loss"].item()
                            n += 1
                val_ce = total / max(n, 1)
                msg = f"[VAL] step {step_global}  CE {val_ce:.4f}"
                notify(msg)
                if val_ce < ce_best:
                    ce_best = val_ce
                    torch.save(model_in_training.state_dict(), os.path.join(cfg.outdir, f"best_step{step_global}.pt"))
                # 항상 마지막 체크포인트도
                torch.save(model_in_training.state_dict(), os.path.join(cfg.outdir, f"last_step{step_global}.pt"))
                model_in_training.train()

    # 최종 저장
    torch.save(model_in_training.state_dict(), os.path.join(cfg.outdir, "final.pt"))
    meta = {
        "best_ce": ce_best,
        "total_steps": total_steps,
        "config": vars(cfg)
    }
    with open(os.path.join(cfg.outdir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()   
    tok = build_tokenizer()
    cfg = TrainCfg(outdir=args.outdir)
    model = make_model(args.kind)
    
    run_name = cfg.outdir.split("/")[-1]  # 혹은 원하는 이름
    wandb_code.wandb_setup(
        run_name=run_name,
        project=WANDB_PROJECT,
        group="pretrain",                     # 여러 모델을 'pretrain' 그룹으로 묶기
        tags=[args.kind, f"L{cfg.seq_len_warm}-{cfg.seq_len_long}"],
        config=dict(
            model_kind=args.kind,
            hidden_size=model.bert.cfg.hidden_size,
            ffn_factor=model.bert.cfg.ffn_expansion_factor,
            num_layers=model.bert.cfg.num_hidden_layers,
            batch_size=cfg.batch_size, lr=cfg.lr, warmup_ratio=cfg.warmup_ratio,
            seq_len_warm=cfg.seq_len_warm, seq_len_long=cfg.seq_len_long,
            total_steps=cfg.warm_steps + cfg.long_steps,
        )
    )
    
    train(model, tok, cfg)
    
    wandb_code.wandb_close()