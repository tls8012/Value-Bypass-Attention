import torch.nn as nn

import boilerplate_code

from configs import VOCAB_SIZE
# =========================================================
# 3) 모델 팩토리: 여기서만 바꿔 끼우면 됨
# =========================================================
def make_model(kind: str, vocab_size=VOCAB_SIZE) -> nn.Module:
    """
    kind:
      - "baseline_4d"    : FFN=4d, ReLU 없음
      - "v_2d"           : FFN=2d, V에 ReLU
      - "q_3d"           : FFN=3d, Q에 ReLU
      - "kv_4d"         : FFN=4d, KV에 ReLU
      - "kv_4a"         : FFN=4d, KV에 ReLU, learnable alpha
    """
    val, dim = kind.split('_')
    q, k, v, d, a = False, False, False, 4, False
    if 'q' in val:
        q = True
    if 'k' in val:
        k = True
    if 'v' in val:
        v = True
    if 'a' in dim:
        a = True
    d = int(dim[0])
    
    cfg = boilerplate_code.BertMiniConfig(vocab_size=vocab_size, ffn_expansion_factor=d, 
                                          add_relu_to_q=q,add_relu_to_k=k,add_relu_to_v=v,
                                          learnable_alpha=a)

    return boilerplate_code.BertMiniForMLM(cfg).apply(boilerplate_code.apply_bert_init)