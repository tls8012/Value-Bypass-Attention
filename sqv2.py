import os, datetime, numpy as np, datetime
from pathlib import Path
from collections import defaultdict
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, default_data_collator,
    TrainingArguments, Trainer,
)

import evaluate

import wandb

from load_model import load_qa_model
from configs import WANDB_SQUAD, notify, KINDS, MAX_CONCURRENCY, SEEDS

tok = AutoTokenizer.from_pretrained("bert-base-uncased") #

# 1) 학습용: 레이블 생성 포함
def prepare_train_features(examples):
    # 질문+문맥 토큰화 (문맥만 자름)
    tokenized = tok(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   # 문자 오프셋
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    # Fast tokenizer는 각 feature에 대해 sequence_ids(i) 를 제공합니다.
    # 0: 질문, 1: 문맥, None: 특수토큰

    start_positions = []
    end_positions   = []

    for i in range(len(tokenized["input_ids"])):
        # 이 feature가 어느 원본 example에서 왔는지
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        offsets = tokenized["offset_mapping"][i]
        seq_ids = tokenized.sequence_ids(i)  # 길이 L의 리스트(0/1/None)
        cls_index = tokenized["input_ids"][i].index(tok.cls_token_id)

        # 문맥(context) 토큰 구간 인덱스 찾기
        # 첫 번째 1이 context 시작, 마지막 1이 context 끝
        context_start = next(idx for idx, s in enumerate(seq_ids) if s == 1)
        context_end   = len(seq_ids) - 1 - next(idx for idx, s in enumerate(reversed(seq_ids)) if s == 1)

        if len(answers["answer_start"]) == 0:
            # (SQuAD v1에선 거의 없음) no-answer일 때는 CLS로 몰기도 함
            start_positions.append(0)
            end_positions.append(0)
            continue

        # 정답의 문자 범위
        start_char = answers["answer_start"][0]
        end_char   = start_char + len(answers["text"][0])

        # 정답이 이 feature의 context window와 겹치지 않으면, CLS 위치로 보냄
        # (겹치면 토큰 인덱스를 계산)
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # 시작 토큰: start_char가 들어갈 마지막 토큰
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            # 끝 토큰: end_char가 들어갈 첫 토큰 이전(= end_char를 넘지 않는 마지막 토큰)
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

        # 평가 편의를 위해: 질문/특수 토큰의 offset은 None으로 바꿔둠
        tokenized["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    # 원본 example id를 보존하면 후처리에 유리
    tokenized["example_id"] = [examples["id"][idx] for idx in sample_mapping]

    return tokenized

# 2) 평가용: 레이블 없이 offset/example_id만 보존
def prepare_eval_features(examples):
    tokenized = tok(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    # context 외 토큰 offset은 None으로
    for i in range(len(tokenized["input_ids"])):
        offsets = tokenized["offset_mapping"][i]
        seq_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]
    tokenized["example_id"] = [examples["id"][idx] for idx in sample_mapping]
    return tokenized

def postprocess_qa_predictions(
    raw_examples, features, raw_predictions,
    n_best_size=40, max_answer_length=30,
    null_score_diff_threshold=0.0  # >0이면 no-answer가 더 쉽게 선택됨
):
    start_logits, end_logits = raw_predictions  # [N_feat, L], [N_feat, L]

    # example_id -> list of feature indices
    feat_by_example = defaultdict(list)
    for i, feat in enumerate(features):
        feat_by_example[feat["example_id"]].append(i)

    predictions, references = [], []

    for ex in raw_examples:
        ex_id = ex["id"]
        feat_indices = feat_by_example.get(ex_id, [])
        best_non_null = {"text": "", "score": -1e30}
        best_null = {"text": "", "score": -1e30}  # 빈 문자열 후보

        for fi in feat_indices:
            s_log = start_logits[fi]
            e_log = end_logits[fi]
            offsets = features[fi]["offset_mapping"]  # context 외는 None
            input_ids = features[fi]["input_ids"]

            # CLS 위치(= null 후보 점수)
            cls_index = input_ids.index(tok.cls_token_id)
            null_score = float(s_log[cls_index] + e_log[cls_index])
            if null_score > best_null["score"]:
                best_null = {"text": "", "score": null_score}

            # context 토큰 인덱스만 남기기
            valid_idx = np.array([i for i, o in enumerate(offsets) if o is not None])
            if valid_idx.size == 0:
                continue

            s_scores = s_log[valid_idx]
            e_scores = e_log[valid_idx]
            s_top = valid_idx[np.argsort(s_scores)[-n_best_size:][::-1]]
            e_top = valid_idx[np.argsort(e_scores)[-n_best_size:][::-1]]

            for s in s_top:
                for e in e_top:
                    if s <= e and (e - s + 1) <= max_answer_length:
                        start_char, end_char = offsets[s][0], offsets[e][1]
                        if start_char is None or end_char is None:
                            continue
                        text = ex["context"][start_char:end_char]
                        score = float(s_log[s] + e_log[e])
                        if score > best_non_null["score"]:
                            best_non_null = {"text": text, "score": score}

        # null vs non-null 비교
        # (원 논문/레퍼런스에선 null_score - best_non_null_score > threshold 면 null)
        score_diff = best_null["score"] - best_non_null["score"]
        if score_diff > null_score_diff_threshold:
            pred_text = ""
            no_ans_prob = 1.0 / (1.0 + np.exp(-(score_diff)))  # 간단 확률 매핑
        else:
            pred_text = best_non_null["text"]
            no_ans_prob = 1.0 / (1.0 + np.exp(-(score_diff)))  # 함께 기록(0~1)

        predictions.append({"id": ex_id, "prediction_text": pred_text, "no_answer_probability": float(no_ans_prob)})
        references.append({"id": ex_id, "answers": ex["answers"]})

    return predictions, references

notify(f"SQUAD init {datetime.datetime.now()}")

def do_this(k, SEED):
    print(k, SEED, os.path.join("./models", f"ckpts_{k}"))
    # 2) 모델 로딩(프리트레인 인코더 + QA 헤드)
    model, report = load_qa_model(os.path.join("./models", f"ckpts_{k}"), tokenizer=tok, map_location="cpu")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    RUN_NAME = f"v2_{k}_{SEED}"
    notify(RUN_NAME+f" Start {datetime.datetime.now()}")
    ds = load_dataset("squad_v2")
    #first100_ids = set(ds["train"]["id"][:100]) # 오버피팅용
    tokenized_ds = ds.map(
        prepare_train_features,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )
    #train_small = tokenized_ds["train"].filter(lambda x: x["example_id"] in first100_ids) # 오버
    # Trainer용 데이터셋(텐서로 못 바꾸는 필드 제거)
    train_for_trainer = tokenized_ds["train"].remove_columns(["offset_mapping", "example_id"]) # tokenized_ds["train"]
    eval_features     = tokenized_ds["validation"]  # ← 후처리에 쓸 원본(그대로 유지)
    eval_for_trainer  = eval_features.remove_columns(["offset_mapping", "example_id"])
    # compute_metrics에서 이걸 참조
    features_for_postproc = eval_features
    # 정답 비율 체킹
    sp = np.array(tokenized_ds["train"]["start_positions"])
    print("pos_window_ratio =", (sp != -100).mean())
    squad_metric = evaluate.load("squad_v2")
    # 5) Trainer 설정
    args = TrainingArguments(
        output_dir=f"./squad_runs/{k}",
        report_to=["wandb"],
        run_name=RUN_NAME,
        seed=SEED,
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available() and (os.environ.get("BF16", "1") != "0"),
        remove_unused_columns=False,   # ⚠️ 커스텀 모델 forward 호환
    )
    
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple) and len(preds)==2:
            start_logits, end_logits = preds
        else:
            start_logits, end_logits = np.split(preds, 2, axis=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits   = end_logits.squeeze(-1)

        # features는 Arrow 형식이라 list로 풀어야 numpy 인덱싱 쉬움
        # features_for_postproc 는 dataset이라 이렇게 한다던데
        feats = [features_for_postproc[i] for i in range(len(features_for_postproc))]
        preds, refs = postprocess_qa_predictions(ds["validation"], feats, (start_logits, end_logits))
        return squad_metric.compute(predictions=preds, references=refs)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_for_trainer,
        eval_dataset=eval_for_trainer,
        tokenizer=tok,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    wandb.init(project=WANDB_SQUAD, name=RUN_NAME, group=f"v2_{k}")
    trainer.train()
    metrics = trainer.evaluate()
    wandb.finish()
    notify(RUN_NAME+f" End {datetime.datetime.now()}")

import multiprocessing as mp, time

def _worker(kind: str,seed: int, gpu_id: int):
    """
    각 시드/종류 조합 하나당 하나의 프로세스로 실행되는 워커.
    전역 변수 SEED, k 를 세팅해 do_this()를 호출합니다.
    """
    # --- GPU 고정 및 시드 고정 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 위로 가렸으니, 프로세스 안에서는 보이는 장치가 하나(인덱스 0)입니다.
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass  # CPU-only에서도 문제 없이 넘어가도록

    # 메모리 파편화 완화(선택): CUDA 11.4+ 권장
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 실행
    do_this(kind, seed)

def _launch_all():
    # 모든 조합 생성
    tasks = [(kind, seed) for kind in KINDS for seed in SEEDS]

    active = []
    idx = 0
    while idx < len(tasks) or active:
        # 빈 자리가 있으면 새 작업 시작
        while idx < len(tasks) and len(active) < MAX_CONCURRENCY:
            kind, seed = tasks[idx]
            p = mp.Process(target=_worker, args=(kind, seed, 0), daemon=False)
            p.start()
            active.append(p)
            idx += 1

        # 어떤 프로세스가 끝났는지 체크
        alive = []
        for p in active:
            if p.is_alive():
                alive.append(p)
            else:
                p.join()
        active = alive

        # 너무 바쁘지 않게 잠깐 쉼
        time.sleep(1)

if __name__ == "__main__":
    # 멀티프로세싱 시작 방식: spawn 권장 (특히 Windows/Mac)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 이미 설정되어 있을 수 있음
        pass

    # 메인 런치
    _launch_all()