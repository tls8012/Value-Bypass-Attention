# psp_eval.py
import math, torch, numpy as np
from typing import List
from torch.nn.functional import cross_entropy

@torch.no_grad()
def pseudo_perplexity(
    model, tokenizer, texts: List[str],
    max_len: int = 512,
    chunk_positions: int = 64,   # 한 번에 마스킹할 위치 수 (메모리에 맞춰 조절)
    device: str = "cuda",
    use_bf16: bool = True
):
    """
    반환: 전체 텍스트에 대한 PSP (낮을수록 좋음)
    전제: model은 MLM 로스/로짓을 낼 수 있어야 함 (MaskedLM)
    """
    model.eval().to(device)
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "Tokenizer must have a [MASK] token"

    total_nll = 0.0
    total_tokens = 0
    
    sent_sum_nll = []    # 문장별 토큰 합 NLL
    sent_len     = []    # 문장별 유효 토큰 수(스페셜/패딩 제외)

    min_tokens = 16

    amp_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else None

    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        )
        if enc["input_ids"].size(1) <= min_tokens:
            continue
        input_ids = enc["input_ids"].to(device)         # [1, L]
        attn      = enc.get("attention_mask", None)
        if attn is not None: attn = attn.to(device)

        L = input_ids.size(1)
        # 마스크 가능한 위치(패딩/CLS/SEP 제외)
        maskable = torch.ones(L, dtype=torch.bool, device=device)
        # 특수 토큰 인덱스 추출(있으면 제외)
        special_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id])
        for i in range(L):
            if int(input_ids[0, i]) in special_ids:
                maskable[i] = False
        pos = torch.nonzero(maskable, as_tuple=False).view(-1)
        if pos.numel() == 0:
            continue
        
        # --- 이 문장에 한정된 누적: 청크들을 합쳐서 '문장 평균'을 만들 것 ---
        sum_nll_s = 0.0
        tok_s = 0
        
        # 위치들을 chunk 단위로 나눠서 처리
        for start in range(0, pos.numel(), chunk_positions):
            p = pos[start:start+chunk_positions]  # [Bpos]
            B = p.numel()
            if B==0:
                continue
            # B개의 샘플 복제
            ids = input_ids.repeat(B, 1)          # [B, L]
            if attn is not None:
                am = attn.repeat(B, 1)            # [B, L]
            else:
                am = None

            # 라벨은 -100으로 채우고, 각 샘플의 해당 위치만 정답으로 설정
            labels = torch.full_like(ids, -100)
            labels[torch.arange(B), p] = ids[0, p]

            # 입력을 마스크로 교체
            ids[torch.arange(B), p] = mask_id

            # forward
            if amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = model(input_ids=ids, attention_mask=am, labels=labels)
                    # 대부분의 MaskedLM은 loss를 스칼라 평균으로 반환
                    loss = out["loss"] if isinstance(out, dict) else out.loss
            else:
                out = model(input_ids=ids, attention_mask=am, labels=labels)
                loss = out["loss"] if isinstance(out, dict) else out.loss

            # 여기서 loss는 '샘플당 1 토큰'이라 batch 평균임 → 합산하려면 B를 곱해줌
            total_nll += float(loss) * B
            total_tokens += B
            # 이 청크에서 마스킹한 토큰이 B개 → 합산하려면 B를 곱한다
            chunk_sum = float(loss) * B
            sum_nll_s += chunk_sum
            tok_s += B
        # 문장 s의 평균 NLL 확보
        if tok_s > 0:
            sent_sum_nll.append(sum_nll_s)
            sent_len.append(tok_s)
    
    nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(nll)
    rt = {"ppl":ppl, "nll":nll}
    
    # 신뢰구간 계산, 95%, 정규분포 가정
    sent_sum_nll = np.asarray(sent_sum_nll, dtype=np.float64)
    sent_len = np.asarray(sent_len, dtype=np.int64)
    lbar = sent_sum_nll / sent_len                 # 문장별 평균 NLL
    w = sent_len / sent_len.sum()                  # 토큰 가중치
    # 가중 평균이 mean_nll와 거의 같아야 함
    mean_ll_check = float((w * lbar).sum())
    # 가중 분산의 불편추정치로 표준오차 계산
    w2_sum = float((w**2).sum())
    sigma2_unb = float((w * (lbar - mean_ll_check)**2).sum() / max(1e-12, (1.0 - w2_sum)))
    se = math.sqrt(sigma2_unb * w2_sum + 1e-12)    # Var(weighted mean) = σ^2 * Σw^2
    ci_ll = (nll - 1.96*se, nll + 1.96*se)
    ci95_psp = (math.exp(ci_ll[0]), math.exp(ci_ll[1]))
    rt['ci_ll'] = ci_ll
    rt['ci_psp'] = ci95_psp
    return rt
