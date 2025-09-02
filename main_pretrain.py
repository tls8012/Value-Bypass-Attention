import subprocess, os
import requests, datetime

from configs import KINDS, notify

from make_dataset import map_dataset

try:
    map_dataset()
    notify(f"Mapping ALL Done {datetime.datetime.now()}")
except Exception as e:
    notify(f"Failed Mapping {datetime.datetime.now()}")
    notify(f"{str(e)}")
    with open(os.path.join("logs", "error.txt"), "a") as f:
        f.write(str(e))
    exit()
    
for k in KINDS:
    try:
        subprocess.run(["python","train_run.py","--kind",k,"--outdir",f"./models/ckpts_{k}"], check=True)
        notify(f"{k} Pretraining Done {datetime.datetime.now()}")
    except subprocess.CalledProcessError as e:
        notify(f"{k} Failed Check {datetime.datetime.now()} \n  (code={e.returncode})")
        with open(os.path.join("logs", "error.txt"), "a") as f:
            f.write(str(e))
notify(f"Pretraining Done {datetime.datetime.now()}")