import torch
from transformers import AutoTokenizer

from load_model import load_mlm_model
from configs import KINDS

def assert_aligned(model, tokenizer):
    V_tok = tokenizer.vocab_size
    emb_w = model.bert.embeddings.word_embeddings.weight
    dec_w = model.decoder.weight
    assert dec_w.data_ptr() == emb_w.data_ptr(), "decoder weight is NOT tied to embeddings!"

# 사용 예시
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
for k in KINDS:
    model = load_mlm_model(f"./models/ckpts_{k}", tokenizer)[0]
    assert_aligned(model, tokenizer)
