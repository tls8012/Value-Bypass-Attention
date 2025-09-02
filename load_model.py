# load_mlm_from_ckpt.py
import os, json, torch
from typing import Tuple
import torch.nn as nn
from boilerplate_code import BertMiniConfig, BertMiniForMLM, BertMini

from configs import VOCAB_SIZE
# 만약 PreTraining 클래스/이름이 다르면 그걸로 교체

def find_ckpt_file(ckpt_dir: str) -> str:
    # 우선순위: final.pt → best_step*.pt → 아무 .pt
    cand = [
        os.path.join(ckpt_dir, "final.pt"),
        *sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("best_step") and f.endswith(".pt")]),
        *sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No .pt found in {ckpt_dir}")

def load_mlm_model(ckpt_dir: str, tokenizer, map_location="cpu") -> Tuple[BertMiniForMLM, dict]:
    """
    ckpt_dir: 프리트레인 저장 폴더(예: ./ckpts_v_relu_4d)
    tokenizer: 프리트레인과 '같은' 토크나이저 객체 (vocab_size 동기화에 사용)
    반환: (모델, load_report)
    """
    m, d = os.path.basename(ckpt_dir)[6:].split('_')
    cfg = BertMiniConfig(vocab_size=VOCAB_SIZE,
                         ffn_expansion_factor= int(d[0]),
                         add_relu_to_q='q' in m,
                         add_relu_to_k='k' in m,
                         add_relu_to_v='v' in m,
                         learnable_alpha='a' in d)

    model = BertMiniForMLM(cfg)
    sd = torch.load(find_ckpt_file(ckpt_dir), map_location=map_location)

    # 로드 & 리포트
    missing, unexpected = model.load_state_dict(sd, strict=False)
    report = {"missing": missing, "unexpected": unexpected}
    if len(missing) > 20 or len(unexpected) > 0:
        print("[WARN] many missing/unexpected keys:",
              len(missing), "missing,", len(unexpected), "unexpected")

    return model, report

class BertMiniForQuestionAnswering(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.cfg = cfg
        self.bert = BertMini(cfg)
        self.qa_outputs = nn.Linear(cfg.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, start_positions=None,end_positions=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # [B,L,H]
        x = self.dropout(x)
        logits = self.qa_outputs(x)                     # [B,L,2]
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss   = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        # loss는 Trainer가 postprocess 후 metric에서 계산하므로 생략(필요시 추가 가능)
        out = {"start_logits": start_logits, "end_logits": end_logits, "loss": total_loss}
        return out
    
def load_qa_model(ckpt_dir: str, tokenizer, map_location="cpu") -> Tuple[nn.Module, dict]:
    """
    프리트레인 MLM ckpt(.pt)를 인코더에 로드하고 QA head를 얹어 반환.
    """
    m, d = os.path.basename(ckpt_dir)[6:].split('_')
    cfg = BertMiniConfig(vocab_size=VOCAB_SIZE,
                        ffn_expansion_factor= int(d[0]),
                        add_relu_to_q='q' in m,
                        add_relu_to_k='k' in m,
                        add_relu_to_v='v' in m,
                        learnable_alpha='a' in d)
    if cfg.max_position_embeddings < 512:
        cfg.max_position_embeddings = 512

    model = BertMiniForQuestionAnswering(cfg)
    sd = torch.load(find_ckpt_file(ckpt_dir), map_location=map_location)

    # 키가 "bert.*" 또는 인코더 레이어인 것만 엄격히 로드(헤드는 랜덤 초기화)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    report = {"missing": missing, "unexpected": unexpected}
    if len(unexpected) > 0:
        print("[load_qa_model][WARN] unexpected:", unexpected[:5], "...")
    if len(missing) > 50:
        print("[load_qa_model][WARN] many missing keys:", len(missing))

    return model, report