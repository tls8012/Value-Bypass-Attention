# logging_wandb.py
import os, wandb, time

def wandb_setup(run_name: str, config: dict, project: str = "bert-mini-pretrain",
                mode: str | None = None, group: str | None = None, tags: list[str] | None = None):
    """
    run_name: 화면에 보일 러닝 이름 (예: "v_relu_2d_0p5B_L128-512")
    config: 하이퍼파라미터/메타 (dict)
    mode: None(기본=online) | "offline" | "disabled"
    group: 여러 run을 한 뭉치(그룹)로 묶어 비교
    tags: ["v_relu","2d","A100"] 같은 라벨
    """
    if mode is None:
        mode = os.environ.get("WANDB_MODE", "online")  # 필요시 export로 바꿀 수 있음
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project=project, name=run_name, mode=mode, group=group, tags=tags)
    if config: wandb.config.update(config, allow_val_change=True)

def wandb_log_train(step: int, loss: float, lr: float, batch_tokens: int, step_sec: float):
    wandb.log({
        "train/loss": float(loss),
        "lr": float(lr),
        "train/tokens_per_step": batch_tokens,
        "train/step_time_sec": step_sec,
        "train/tokens_per_sec": batch_tokens / max(step_sec, 1e-6),
    }, step=step)

def wandb_log_val(step: int, val_ce: float):
    wandb.log({"val/ce": float(val_ce)}, step=step)

def wandb_log_general(msg, step):
    wandb.log(msg, step=step)

def wandb_close():
    wandb.finish()
