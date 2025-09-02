import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)

from configs import DATASET_PATH, VOCAB_SIZE, HIDDEN_LAYERS, HIDDEN_DIM, NONLIN_ALPHA, HEADS

class BertMiniConfig:
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_DIM,
        num_hidden_layers=HIDDEN_LAYERS,
        num_attention_heads=HEADS,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ffn_expansion_factor=4,   # 2, 3, 4
        add_relu_to_q=False,
        add_relu_to_k=False,
        add_relu_to_v=False,      # baseline=False, V실험=True
        nonlin_alpha=NONLIN_ALPHA,
        learnable_alpha=False,
        layer_norm_eps=1e-6,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_expansion_factor = ffn_expansion_factor
        self.add_relu_to_q = add_relu_to_q
        self.add_relu_to_k = add_relu_to_k
        self.add_relu_to_v = add_relu_to_v
        self.nonlin_alpha = nonlin_alpha
        self.learnable_alpha = learnable_alpha
        self.layer_norm_eps = layer_norm_eps
        
class BertEmbeddings(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        bsz, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = input_ids.new_zeros(input_ids.shape)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.word_embeddings(input_ids) \
            + self.position_embeddings(pos_ids) \
            + self.token_type_embeddings(token_type_ids)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x

import math

def sdpa_safe(q, k, v, key_padding_mask_bool, dropout_p, training):
    # q,k,v: [B, h, L, Dh]
    B, H, L, Dh = q.shape
    # bool: True=mask(가리기). float로 변환해 -inf 추가
    if key_padding_mask_bool is not None:
        # [B,1,1,L] -> [B, H, L, L] 로 브로드캐스트
        pad = key_padding_mask_bool  # True=mask
        attn_mask = pad.to(q.dtype) * (-1e9)  # True -> -1e9, False -> 0
        attn_mask = attn_mask.expand(B, H, L, L)
    else:
        attn_mask = None

    scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)  # 수동 스케일
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = scores.softmax(dim=-1)
    if training and dropout_p > 0:
        probs = torch.nn.functional.dropout(probs, p=dropout_p)
    return probs @ v

class BertSelfAttention(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size % num_heads != 0")
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

        if cfg.learnable_alpha:
            self.alpha_q = nn.Parameter(torch.tensor(cfg.nonlin_alpha)) if cfg.add_relu_to_q else None
            self.alpha_k = nn.Parameter(torch.tensor(cfg.nonlin_alpha)) if cfg.add_relu_to_k else None
            self.alpha_v = nn.Parameter(torch.tensor(cfg.nonlin_alpha)) if cfg.add_relu_to_v else None
        else:
            self.register_buffer("alpha_q_const", torch.tensor(cfg.nonlin_alpha))
            self.register_buffer("alpha_k_const", torch.tensor(cfg.nonlin_alpha))
            self.register_buffer("alpha_v_const", torch.tensor(cfg.nonlin_alpha))
            self.alpha_q = self.alpha_k = self.alpha_v = None

    def _add_relu(self, t, x, which):
        if which == "q" and self.cfg.add_relu_to_q:
            a = self.alpha_q if self.alpha_q is not None else self.alpha_q_const
            t = t + a * F.relu(x)
        elif which == "k" and self.cfg.add_relu_to_k:
            a = self.alpha_k if self.alpha_k is not None else self.alpha_k_const
            t = t + a * F.relu(x)
        elif which == "v" and self.cfg.add_relu_to_v:
            a = self.alpha_v if self.alpha_v is not None else self.alpha_v_const
            t = t + a * F.relu(x)
        return t

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H]
        b, l, h = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self._add_relu(q, hidden_states, "q")
        k = self._add_relu(k, hidden_states, "k")
        v = self._add_relu(v, hidden_states, "v")

        # [B, L, H] -> [B, heads, L, Dh]
        def shape(x):
            return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(q); k = shape(k); v = shape(v)

        # PyTorch SDPA (FlashAttention2로 자동 라우팅될 수 있음)
        """attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,   # [B, 1, L, L] or [B*heads, L, L]도 허용
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )"""
        attn = sdpa_safe(q, k, v, attention_mask, self.dropout.p, self.training)
        # [B, heads, L, Dh] -> [B, L, H]
        attn = attn.transpose(1, 2).contiguous().view(b, l, h)
        out = self.out_proj(attn)
        return out

class BertFFN(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        inter = cfg.hidden_size * cfg.ffn_expansion_factor
        self.dense_in = nn.Linear(cfg.hidden_size, inter)
        self.act = nn.GELU()
        self.dense_out = nn.Linear(inter, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.act(x)
        x = self.dense_out(x)
        x = self.dropout(x)
        return x

class BertLayer(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.attn = BertSelfAttention(cfg)
        self.attn_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = BertFFN(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x, attention_mask=None):
        # POST-LN
        """a = self.attn(x, attention_mask=attention_mask)
        x = self.attn_norm(x + self.attn_dropout(a))
        f = self.ffn(x)
        x = self.ffn_norm(x + f)"""
        # Pre-LN
        a = self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.attn_dropout(a)
        f = self.ffn(self.ffn_norm(x))
        x = x + f
        return x

class BertEncoder(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x

class BertMini(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.cfg = cfg

    def build_padding_mask(self, attention_mask_bool):
        # boolean [B, L] (1=keep, 0=pad) -> SDPA용 [B, 1, L, L] additive mask가 아님
        # SDPA는 float mask도 받지만, 여기선 [B, 1, L, L] 형태의 bool mask 사용
        b, l = attention_mask_bool.shape
        # key쪽에만 패딩을 가리기: [B, 1, 1, L] 형태가 일반적
        return (~attention_mask_bool.bool()).unsqueeze(1).unsqueeze(2)  # True=mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        x = self.embeddings(input_ids, token_type_ids)
        sdpa_mask = None
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            # PyTorch SDPA는 bool mask에서 True=mask로 해석함
            sdpa_mask = self.build_padding_mask(attention_mask)  # [B,1,1,L]
        x = self.encoder(x, attention_mask=sdpa_mask)
        return x

class BertMiniForMLM(nn.Module):
    def __init__(self, cfg: BertMiniConfig):
        super().__init__()
        self.bert = BertMini(cfg)
        #self.logit_scale = nn.Parameter(torch.tensor(1.0))  # 학습 가능 온도
        self.lm_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=True)
        self.decoder.weight = self.bert.embeddings.word_embeddings.weight  # tie

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        x = self.lm_head(x)
        logits = self.decoder(x) #/ self.logit_scale.clamp_min(0.5)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return {"loss": loss, "logits": logits}

# =======
# BERT Style Init
# model.apply(apply_bert_init)
def apply_bert_init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # ★ BERT std
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


# =========================================================
# 2) 데이터/토크나이저/콜레이터
# =========================================================
def build_tokenizer(save_dir="./bert_base_uncased"):
    # 편의상 공개 토크나이저를 씀 (실험 공정성 위해 '동일 토크나이저' 유지가 중요)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 필요 시 아래처럼 저장해서 고정
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tok.save_pretrained(save_dir)
    return tok


def load_corpus(
    tokenizer: PreTrainedTokenizerFast,
    seq_len: int = 128,
    val_split: float = 0.02,
    mode: str = "cached_map",   # "set_transform" | "cached_map"
):
    """
    mode="set_transform":  캐싱 없이 즉시 시작(권장, 코랩 T4용 빠른 스타트)
    mode="cached_map":     한 번 캐시 만들어 다음 세션에서 재사용
    """
    # 2) on-the-fly 토큰화 (빠름: 디스크 캐시 X)
    if mode == "set_transform":
        # 1) 위키 로드 & 분할
        ds = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
        ds = ds.train_test_split(test_size=val_split, seed=42)
        ds_train, ds_val = ds["train"], ds["test"]
        def _transform(batch):
            # batched=True처럼 여러 샘플 처리
            tok = tokenizer(
                batch["text"],
                truncation=True,
                max_length=seq_len,
                padding=False,      # collator가 패딩/MLM 처리
            )
            return tok
        ds_train.set_transform(_transform)
        ds_val.set_transform(_transform)
        # 원래 컬럼(text)은 남아있어도 상관없음(불러올 때 transform 결과가 우선)
        return ds_train, ds_val

    # 3) 캐시된 데이터 재사용 (위치에 존재한다고 가정)
    elif mode == "cached_map":
        ds_train = os.path.join(DATASET_PATH, str(seq_len), "train")
        ds_val   = os.path.join(DATASET_PATH, str(seq_len), "val")
        if not os.path.exists(ds_train) or not os.path.exists(ds_val):
            raise ValueError(f"Dataset Not Found: {seq_len}")
        ds_train = load_from_disk(ds_train)
        ds_val = load_from_disk(ds_val)
        return ds_train, ds_val

    else:
        raise ValueError(f"Unknown mode: {mode}")
def build_collator(tokenizer, mlm_probability=0.15):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
