"""RepCounter backend: count push-up reps from a video (YOLOv8n-pose + elbow angle)."""
import csv
import os
import queue
import threading
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

EXERCISE = "pushup"
START_T = 130   # elbow angle drops below this -> a rep starts
END_T = 150     # elbow angle rises above this -> the rep ends
FULL_T = 100    # lowest angle must go below this for a full rep
MIN_DROP = 120  # ignore small wiggles that never go below this
SMOOTH = 5      # frames in the moving average
MIN_CONF = 0.3  # minimum keypoint confidence
WRITE = os.environ.get("RC_NOVIDEO") != "1"
SIDES = {"left": (5, 7, 9), "right": (6, 8, 10)}  # shoulder, elbow, wrist
LABEL = "Push-ups"
ELBOW = {"left": (5, 7, 9), "right": (6, 8, 10)}      # shoulder, elbow, wrist
KNEE = {"left": (11, 13, 15), "right": (12, 14, 16)}  # hip, knee, ankle
EXERCISES = {
    "pushup": dict(sides=ELBOW, start=130, end=150, full=100, drop=120, label="Push-ups"),
    "curl": dict(sides=ELBOW, start=120, end=135, full=70, drop=100, label="Curls"),
    "squat": dict(sides=KNEE, start=150, end=160, full=100, drop=140, label="Squats"),
}


def set_exercise(name):
    """Load the joints and thresholds of one exercise."""
    global EXERCISE, SIDES, START_T, END_T, FULL_T, MIN_DROP, LABEL
    c = EXERCISES[name]
    EXERCISE, SIDES, LABEL = name, c["sides"], c["label"]
    START_T, END_T, FULL_T, MIN_DROP = c["start"], c["end"], c["full"], c["drop"]


def angle(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def pick_person(res):
    """Index of the largest person in the frame."""
    if res.boxes is None or len(res.boxes) == 0:
        return None
    wh = res.boxes.xywh.cpu().numpy()
    return int(np.argmax(wh[:, 2] * wh[:, 3]))


def elbow_angle(xy, conf):
    """Elbow angle of the side whose 3 keypoints are most confident."""
    best = None
    for s, e, w in SIDES.values():
        c = min(conf[s], conf[e], conf[w])
        if c >= MIN_CONF and (best is None or c > best[0]):
            best = (c, angle(xy[s], xy[e], xy[w]))
    return None if best is None else best[1]


def main(src, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    model = YOLO(os.environ.get("RC_MODEL", "yolov8n-pose.pt"), task="pose")
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    OW = min(W, 960)
    OH = int(H * OW / W) // 2 * 2
    writer = cv2.VideoWriter(f"{out_dir}/{name}_annotated.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
    q = queue.Queue(maxsize=16)

    def write_loop():
        while True:
            im = q.get()
            if im is None:
                break
            writer.write(cv2.resize(im, (OW, OH)))

    th = threading.Thread(target=write_loop, daemon=True)
    th.start()

    hist = deque(maxlen=SMOOTH)
    state, start_f, min_a, top_max = "UP", 0, 999.0, 0.0
    reps, angles = [], []
    f, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          verbose=False, imgsz=640)[0]
        ang = None
        i = pick_person(res)
        if i is not None and res.keypoints is not None:
            xy = res.keypoints.xy[i].cpu().numpy()
            conf = res.keypoints.conf[i].cpu().numpy()
            raw = elbow_angle(xy, conf)
            if raw is not None:
                hist.append(raw)
                ang = float(np.mean(hist))

        if ang is not None:
            if state == "UP":
                top_max = max(top_max, ang)
                if ang < START_T:
                    state, start_f, min_a = "DOWN", f, ang
            else:
                min_a = min(min_a, ang)
                if ang > END_T:
                    state = "UP"
                    if min_a < MIN_DROP:
                        reps.append({
                            "rep": len(reps) + 1, "exercise": EXERCISE,
                            "start_s": round(start_f / fps, 2),
                            "end_s": round(f / fps, 2),
                            "duration_s": round((f - start_f) / fps, 2),
                            "min_angle": round(min_a),
                            "max_angle": round(top_max),
                            "quality": "full" if min_a < FULL_T else "partial"})
                    top_max = ang
        angles.append((f, round(f / fps, 3), None if ang is None else round(ang, 1)))

        out = res.plot(boxes=False)
        k = max(1.0, W / 640)  # scale the text with the video size
        cv2.rectangle(out, (0, 0), (int(330 * k), int(90 * k)), (0, 0, 0), -1)
        cv2.putText(out, f"{LABEL}: {len(reps)}", (int(10 * k), int(40 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * k, (0, 255, 0), int(3 * k))
        txt = "angle: --" if ang is None else f"angle: {ang:.0f}  {state}"
        cv2.putText(out, txt, (int(10 * k), int(75 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (255, 255, 255), int(2 * k))
        if WRITE:
            q.put(out)
        f += 1

    cap.release()
    q.put(None)
    th.join()
    writer.release()
    secs = time.time() - t0

    with open(f"{out_dir}/{name}_reps.csv", "w", newline="") as fh:
        cols = ["rep", "exercise", "start_s", "end_s", "duration_s",
                "min_angle", "max_angle", "quality"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(reps)
    with open(f"{out_dir}/{name}_angles.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "angle"])
        wr.writerows(angles)

    full = sum(r["quality"] == "full" for r in reps)
    avg = np.mean([r["duration_s"] for r in reps]) if reps else 0
    print(f"{name}: total={len(reps)} full={full} partial={len(reps) - full} "
          f"avg_s/rep={avg:.2f}  FPS={f / secs:.1f}")


if __name__ == "__main__":
    set_exercise(sys.argv[2] if len(sys.argv) > 2 else "pushup")
    main(sys.argv[1])
