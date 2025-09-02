import os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

import wandb

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer, TrainingArguments,
)

# === 1) 네 커스텀 인코더 가져오기 ===
from boilerplate_code import BertMini, BertMiniConfig  # <- 경로 맞추기

# === 2) 분류용 헤드 래퍼 ===
class BertMiniForSequenceClassification(nn.Module):
    def __init__(self, cfg: BertMiniConfig, num_labels: int, problem_type: str = "single_label_classification"):
        super().__init__()
        self.bert = BertMini(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)
        self.num_labels = num_labels
        self.problem_type = problem_type  # "single_label_classification" | "regression"
        self.cfg = cfg

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # [B, L, H] -> [B, H] (단순 CLS pooling: 첫 토큰 사용)
        pooled = x[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                logits = logits.view(-1)
                loss = F.mse_loss(logits, labels.view(-1))
            else:
                loss = F.cross_entropy(logits, labels.view(-1))
        return {"loss": loss, "logits": logits}

    # Trainer 호환(선택적): 토크나이저 weight tying 등에서 사용
    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

# === 3) 태스크 스펙 ===
GLUE_TASKS = {
    # name: (num_labels, metric_name(s), is_regression, default_max_len)
    "sst2":  (2, ["accuracy"], False, 128),
    "mrpc":  (2, ["accuracy","f1"], False, 128),
    "cola":  (2, ["matthews_correlation"], False, 128),
    "qnli":  (2, ["accuracy"], False, 256),
    "rte":   (2, ["accuracy"], False, 256),
    "qqp":   (2, ["accuracy","f1"], False, 128),
    "mnli":  (3, ["accuracy"], False, 384),   # matched
    "stsb":  (1, ["pearson","spearmanr"], True, 256),
}

def build_model_from_ckpt(ckpt_path: str, num_labels: int, is_regression: bool, vocab_size: int):
    cfg = BertMiniConfig(vocab_size=vocab_size)
    model = BertMiniForSequenceClassification(
        cfg, num_labels=num_labels,
        problem_type=("regression" if is_regression else "single_label_classification")
    )
    if ckpt_path and os.path.exists(ckpt_path):
        # 프리트레인(MLM)에서 encoder만 로드하고 헤드는 무시
        sd = torch.load(ckpt_path, map_location="cpu")
        # 우리가 저장했던 키와 다르면 strict=False로 로드
        model.load_state_dict(sd, strict=False)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=GLUE_TASKS.keys())
    ap.add_argument("--ckpt", required=False, default="", help="pretrained .pt (strict=False로 로드)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)       # 미니모델 추천 범위: 2e-4 ~ 1e-3
    ap.add_argument("--max_len", type=int, default=0)       # 0이면 태스크 기본값 사용
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    task = args.task
    num_labels, metric_names, is_reg, default_max_len = GLUE_TASKS[task]
    max_len = args.max_len or default_max_len

    model_name = os.path.basename(os.path.dirname(args.ckpt))[6:]
    
    wandb.init(project="gluev", name=f"{model_name}_{task}_{args.seed}", group=f"{model_name}_{task}")
    
    # 1) 데이터/토크나이저
    raw = load_dataset("glue", "mnli" if task=="mnli" else task)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)  # 프리트레인과 동일

    sentence1_key, sentence2_key = {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp":  ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte":  ("sentence1", "sentence2"),
    }[task]

    def preprocess(ex):
        if sentence2_key is None:
            return tok(ex[sentence1_key], truncation=True, max_length=max_len)
        else:
            return tok(ex[sentence1_key], ex[sentence2_key], truncation=True, max_length=max_len)

    cols_to_rm = list(set(raw["train"].column_names) - {"label"})
    ds = raw.map(preprocess, batched=True, remove_columns=cols_to_rm)

    # STS-B는 label이 float, 나머지는 int
    if is_reg:
        ds = ds.map(lambda x: {"labels": [float(v) for v in x["label"]]}, batched=True)
    else:
        ds = ds.rename_column("label", "labels")

    # MNLI는 두 개의 eval split
    eval_splits = ["validation_matched","validation_mismatched"] if task=="mnli" else ["validation"]

    # 2) 모델
    model = build_model_from_ckpt(args.ckpt, num_labels, is_reg, vocab_size=32000)

    # 3) metrics
    metrics = [evaluate.load("glue", "mnli" if (task=="mnli" and m=="accuracy") else task) for m in metric_names]
    def compute_metrics(p):
        preds = p.predictions
        if is_reg:
            preds = preds.reshape(-1)
            res = {}
            for m in metrics:
                # STS-B: pearson/spearman
                res.update(m.compute(predictions=preds, references=p.label_ids))
            # 편의상 평균도 추가
            res["corr_avg"] = float(np.mean(list(res.values())))
            return res
        else:
            preds = preds.argmax(-1)
            res = {}
            for m in metrics:
                res.update(m.compute(predictions=preds, references=p.label_ids))
            return res

    # 4) Trainer
    collator = DataCollatorWithPadding(tok)
    args_tr = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        metric_for_best_model="accuracy" if not is_reg else "corr_avg",
        greater_is_better=True,
        report_to=["wandb"],  # wandb 쓰려면 ["wandb"]
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds["train"],
        eval_dataset=ds[eval_splits[0]],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # MNLI면 두 eval 다 찍기
    print("== Eval (primary) ==")
    print(trainer.evaluate())
    if task=="mnli":
        print("== Eval (mismatched) ==")
        print(trainer.evaluate(ds[eval_splits[1]]))

    # 최종 저장
    os.makedirs(args.outdir, exist_ok=True)
    #torch.save(model.state_dict(), os.path.join(args.outdir, "finetuned.pt"))

if __name__ == "__main__":
    main()
