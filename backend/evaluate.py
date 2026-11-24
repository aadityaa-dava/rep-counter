"""Run the rep counter on every clip in data/ground_truth.csv and score it."""
import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np

# No annotated video (faster); use the TensorRT engine unless told otherwise
env = dict(os.environ, RC_NOVIDEO="1")
env.setdefault("RC_MODEL", "yolov8n-pose.engine")

rows = []
for g in csv.DictReader(open("data/ground_truth.csv")):
    path = glob.glob(f"data/clips/{g['clip']}.*")[0]
    out = subprocess.run([sys.executable, "backend/rep_counter.py", path, g["exercise"]],
                         capture_output=True, text=True, env=env).stdout
    # parse: "clip: total=3 full=3 partial=0 avg_s/rep=1.11  FPS=16.2"
    m = re.search(r"full=(\d+) partial=(\d+).*FPS=([\d.]+)", out)
    pred, fps = int(m.group(1)), float(m.group(3))   # score = number of FULL reps
    truth = int(g["truth"])
    rows.append(dict(clip=g["clip"], exercise=g["exercise"], split=g["split"],
                     truth=truth, pred=pred, error=pred - truth, fps=fps))
    print(f"{g['clip']:12s} truth={truth:3d} pred={pred:3d} error={pred - truth:+d} fps={fps:.1f}")

# save the per-clip results
os.makedirs("results", exist_ok=True)
with open("results/evaluation.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)


def summary(name, rs):
    """Exact-match rate, mean absolute error, total count accuracy, mean FPS."""
    if not rs:
        return
    err = [abs(r["error"]) for r in rs]
    truth = sum(r["truth"] for r in rs)
    print(f"{name:10s} clips={len(rs):2d}  exact={sum(e == 0 for e in err) / len(rs):.0%}  "
          f"MAE={np.mean(err):.2f}  accuracy={1 - sum(err) / truth:.1%}  "
          f"FPS={np.mean([r['fps'] for r in rs]):.1f}")


print()
summary("tuning", [r for r in rows if r["split"] == "tune"])
summary("held-out", [r for r in rows if r["split"] == "heldout"])
summary("all", rows)
