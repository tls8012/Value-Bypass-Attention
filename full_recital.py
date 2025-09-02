from configs import notify

import subprocess, sys, datetime

def run_stage(project_name: str, module_name: str):
    # 이 코드 문자열은 새 파이썬 프로세스 안에서 실행됩니다.
    code = (
        "import configs;"
        f"configs.WANDB_PROJECT={project_name!r};"
        f"import {module_name};"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
notify(f"Full process Start {datetime.datetime.now()}")

run_stage("augv",  "main_pretrain")  # 기존 main_pretrain.py 그대로 사용
notify(f"PSP Start {datetime.datetime.now()}")

run_stage("pspv",  "main_psp")       # 기존 main_psp.py 그대로 사용
notify(f"SQUAD Start {datetime.datetime.now()}")

run_stage("squadv","squad")          # 기존 squad.py 그대로 사용
notify(f"Full process DONE! {datetime.datetime.now()}")

