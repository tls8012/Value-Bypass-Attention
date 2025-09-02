import torch, requests

DISCORD_URL = "https://discord.com/api/webhooks/"

KINDS = ["6baseline_4d","6v_4d",
         "6baseline_1d", "6v_1d"]
# "baseline_2d", "baseline_3d", "baseline_4d","v_2d","v_3d", "v_4d", "q_2d", "q_3d","q_4d","k_2d", "k_3d", "k_4d"
# "v_2d","v_3d", "v_4d", "v_2a", "v_3a", "v_4a""baseline_4d","v_4d","v_4a","baseline_2d", "baseline_3d", "v_2d","v_3d", "v_2a", "v_3a"
# "baseline_1d", "v_1d", "v_1a"
# "8baseline_4d","8v_4d","8v_4a","8baseline_2d", "8baseline_3d", "8v_2d","8v_3d", "8v_2a", "8v_3a", "8baseline_1d", "8v_1d", "8v_1a"
WARM_STEPS = 65000

LONG_STEPS = 15000

BATCH = 48

SEQ_WARM = 512

SEQ_LONG = 512

LR = 1E-4

HIDDEN_LAYERS = 6

HIDDEN_DIM = 384

HEADS = int(HIDDEN_DIM // 64)

NONLIN_ALPHA = 0.5

GRAD_ACCUM_STEPS = 4

SAVE_EVERY = 25000

LOG_EVERY = 1000

VOCAB_SIZE = 30522 #30522

DATASET_PATH = "./dataset/wiki_en/" # "wiki_en", wikitext_mini

DATASET_NAME = "wikipedia"
DATASET_SETTING = "20220301.en"

WANDB_PROJECT = "augv"
WANDB_PSP = "pspv"
WANDB_SQUAD = "squadv"

SEEDS = [42, 2, 8, 9, 5] # 42 and first 4 numbers from 1 to 9 shuffled 
MAX_CONCURRENCY = 4

# ---- 알림 도우미 ----
def notify(msg: str):
    if DISCORD_URL:
        try: requests.post(DISCORD_URL, json={"content": msg})
        except Exception: pass
    print(msg)

# =========================================================
# 0) 하드웨어/성능 세팅 (A100 권장)
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"