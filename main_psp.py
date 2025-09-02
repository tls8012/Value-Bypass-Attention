import requests, datetime

from datasets import load_dataset
from transformers import AutoTokenizer

import wandb

from load_model import load_mlm_model
from psp import pseudo_perplexity

from configs import KINDS, notify, WANDB_PSP

notify(f"init psp test {datetime.datetime.now()}")

tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# 1) 평가 텍스트 준비 (권장: 프리트레인에 안 쓴 held-out / or wikitext-103 test 일부)
NUM_SENTENCES = 5000
ds = load_dataset("wikitext", "wikitext-103-v1", split=f"test[:{NUM_SENTENCES}]")
texts = [t for t in ds["text"] if len(t.strip()) > 0]

notify(f"loaded psp test {datetime.datetime.now()}")

for k in KINDS:
    notify(f"psp test {k} {datetime.datetime.now()}")
    # 2) 모델 로드
    model, _ = load_mlm_model(f"./models/ckpts_{k}", tokenizer=tok)
    wandb.init(project=WANDB_PSP, name=f"{k}_{NUM_SENTENCES}", group=f"wikitext_103_test_{NUM_SENTENCES}")

    # 3) PSP 계산
    rt = pseudo_perplexity(model, tok, texts, max_len=512, chunk_positions=64, device="cuda", use_bf16=True)
    ppl = rt['ppl']
    nll = rt['nll']
    ci_ll = rt['ci_ll']
    ci_psp = rt['ci_psp']
    print("PSP:", ppl, 'NLL', nll)
    
    wandb.log({f"PSP/{NUM_SENTENCES}": ppl, f'NLL/{NUM_SENTENCES}': nll, 
               f"ci_ll_low/{NUM_SENTENCES}":ci_ll[0], f"ci_ll_hi/{NUM_SENTENCES}": ci_ll[1],
               f"ci_psp_low/{NUM_SENTENCES}": ci_psp[0], f"ci_psp_hi/{NUM_SENTENCES}": ci_psp[1]})
    
    notify(f"psp Done {k} {ppl} {datetime.datetime.now()}")
    
    wandb.finish()
