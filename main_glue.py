import subprocess, os
import requests, datetime

import configs
from configs import KINDS, notify

ROOT_MODELS = "./models/glue"

if not os.path.exists(ROOT_MODELS):
    os.makedirs(ROOT_MODELS)
    
SEEDS = ["42", "1","2","3","4"]

TASK_CFG = {
    "sst2": {"epochs": 3, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "qnli": {"epochs": 3, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "qqp":  {"epochs": 3, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "mnli": {"epochs": 3, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "mrpc": {"epochs": 5, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "cola": {"epochs": 5, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "rte":  {"epochs": 5, "bsz": 64, "lr": 5e-5, "max_len": 128},
    "stsb": {"epochs": 5, "bsz": 64, "lr": 5e-5, "max_len": 128},  # 회귀
}

for k in KINDS:
    for task in TASK_CFG.keys():
        for seed in SEEDS:
            outdir = os.path.join(ROOT_MODELS, k, task, seed)
            os.makedirs(outdir, exist_ok=True)
            glue_test = ["python","glue_train.py",
                         "--task",task,
                         "--ckpt", os.path.join("./models", f"ckpts_{k}", "final.pt"),
                         "--outdir", outdir,]
            glue_test.extend([item for sublist in [["--"+key, str(val)] for key, val in TASK_CFG[task].items()] for item in sublist])
            glue_test.extend(["--seed", seed])
            try:
                subprocess.run(glue_test, check=True)
                notify(f"{k} {task} {seed} Glue Test Done {datetime.datetime.now()}")
            except subprocess.CalledProcessError as e:
                notify(f"{k} Failed Check {datetime.datetime.now()} \n  (code={e.returncode})")
                with open(os.path.join("logs", "error.txt"), "a") as f:
                    f.write(str(e))
            
notify(f"Glue Test Done {datetime.datetime.now()}")