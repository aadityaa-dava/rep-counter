"""RepCounter dashboard: pick an exercise, choose or upload a video, run the counter, review the reps."""
import glob
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, "backend")
from rep_counter import EXERCISES  # thresholds, used only to draw the plot

EXERCISE_NAMES = {"Push-ups": "pushup", "Squats": "squat", "Bicep curls": "curl"}
CLIP_PREFIX = {"pushup": "pushup", "squat": "squat", "curl": "bicepcurl"}
ENGINE = "yolov8n-pose.engine"
TEAL, ORANGE, NAVY = "#0FA37F", "#F26B1D", "#14213D"
OUT = "results"
UPLOADS = "data/uploads"


def run_counter(path, exercise, use_trt):
    """Run backend/rep_counter.py on one video and return the finished process."""
    env = dict(os.environ)
    if use_trt:
        env["RC_MODEL"] = ENGINE
    return subprocess.run([sys.executable, "backend/rep_counter.py", path, exercise],
                          capture_output=True, text=True, env=env)


def web_video(name):
    """Convert the annotated video to H.264 so the browser can play it."""
    src, dst = f"{OUT}/{name}_annotated.mp4", f"{OUT}/{name}_web.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", dst])
    return dst


def angle_plot(name, exercise, reps):
    """Joint angle over time, with the two thresholds and every counted rep shaded."""
    ang = pd.read_csv(f"{OUT}/{name}_angles.csv")
    cfg = EXERCISES[exercise]
    bg, grid, muted = "#FFFFFF", "#E3E9F1", "#5B6B82"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.plot(ang["time_s"], ang["angle"], lw=2.2, color=TEAL)
    ax.axhline(cfg["start"], color="#E5484D", ls="--", lw=1, label=f"rep starts below {cfg['start']}°")
    ax.axhline(cfg["end"], color="#2F7DE1", ls="--", lw=1, label=f"rep ends above {cfg['end']}°")
    for _, r in reps.iterrows():
        color = TEAL if r["quality"] == "full" else ORANGE
        ax.axvspan(r["start_s"], r["end_s"], color=color, alpha=0.16, lw=0)
        ax.text((r["start_s"] + r["end_s"]) / 2, 183, str(int(r["rep"])), ha="center",
                fontsize=9, color=NAVY, fontweight="bold")
    ax.set_ylim(0, 195)
    ax.set_xlabel("time (s)", color=muted)
    ax.set_ylabel("joint angle (degrees)", color=muted)
    ax.tick_params(colors=muted)
    ax.grid(color=grid, alpha=0.5, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.legend(loc="lower right", fontsize=8, facecolor=bg, edgecolor="none", labelcolor=muted)
    fig.tight_layout()
    return fig


def card(label, value, color=NAVY, cls=""):
    """One rounded info card (HTML)."""
    return (f'<div class="card {cls}"><div class="lab">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div></div>')


# Two key poses per exercise (x, y in a 160 x 116 box). The figure moves between them.
POSES = {
    "pushup": dict(floor=96, dur=2.2, track="elbow",
                   a=dict(head=(112, 56), shoulder=(100, 62), elbow=(102, 77), wrist=(104, 92),
                          hip=(62, 76), knee=(43, 83), ankle=(24, 90)),
                   b=dict(head=(112, 76), shoulder=(100, 80), elbow=(86, 86), wrist=(104, 92),
                          hip=(68, 84), knee=(46, 87), ankle=(24, 90))),
    "squat": dict(floor=104, dur=2.6, track="knee",
                  a=dict(head=(60, 12), shoulder=(60, 24), elbow=(70, 28), wrist=(82, 28),
                         hip=(60, 54), knee=(60, 80), ankle=(60, 104)),
                  b=dict(head=(74, 44), shoulder=(68, 54), elbow=(79, 56), wrist=(91, 56),
                         hip=(49, 77), knee=(74, 85), ankle=(60, 104))),
    "curl": dict(floor=104, dur=1.8, track="elbow",
                 a=dict(head=(60, 12), shoulder=(60, 24), elbow=(67, 40), wrist=(69, 57),
                        hip=(60, 54), knee=(60, 80), ankle=(60, 104)),
                 b=dict(head=(60, 12), shoulder=(60, 24), elbow=(67, 40), wrist=(82, 30),
                        hip=(60, 54), knee=(60, 80), ankle=(60, 104))),
}
BONES = [("head", "shoulder"), ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
         ("shoulder", "elbow"), ("elbow", "wrist")]


def exercise_svg(kind):
    """Animated stick-figure skeleton doing one exercise; the measured joint is orange."""
    p = POSES[kind]
    a, b = p["a"], p["b"]

    def anim(attr, v1, v2):
        return (f'<animate attributeName="{attr}" values="{v1};{v2};{v1}" keyTimes="0;0.5;1" '
                f'calcMode="spline" keySplines="0.45 0 0.55 1;0.45 0 0.55 1" '
                f'dur="{p["dur"]}s" repeatCount="indefinite"/>')

    parts = [f'<line x1="8" y1="{p["floor"] + 5}" x2="152" y2="{p["floor"] + 5}" '
             f'stroke="#C9D5E3" stroke-width="2" stroke-linecap="round"/>']
    for j, k in BONES:
        parts.append(
            f'<line x1="{a[j][0]}" y1="{a[j][1]}" x2="{a[k][0]}" y2="{a[k][1]}" stroke="{NAVY}" '
            f'stroke-width="3" stroke-linecap="round">'
            + anim("x1", a[j][0], b[j][0]) + anim("y1", a[j][1], b[j][1])
            + anim("x2", a[k][0], b[k][0]) + anim("y2", a[k][1], b[k][1]) + "</line>")
    for name, pt in a.items():
        r, fill = (8, "none") if name == "head" else (3.6, TEAL)
        stroke = f' stroke="{NAVY}" stroke-width="3"' if name == "head" else ""
        if name == p["track"]:
            r, fill = 5.5, ORANGE
        parts.append(f'<circle cx="{pt[0]}" cy="{pt[1]}" r="{r}" fill="{fill}"{stroke}>'
                     + anim("cx", pt[0], b[name][0]) + anim("cy", pt[1], b[name][1]) + "</circle>")
    return f'<svg viewBox="0 0 160 116" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def steps_html(steps):
    """Vertical flow: step 1, arrow down, step 2, arrow down, step 3."""
    arrow = ('<div class="arrow"><svg width="26" height="30" viewBox="0 0 26 30"><path d="M13 3 V21 '
             'M5 14 L13 23 L21 14" stroke="#0FA37F" stroke-width="3" fill="none" stroke-linecap="round" '
             'stroke-linejoin="round"><animateTransform attributeName="transform" type="translate" '
             'values="0 0;0 3;0 0" dur="1.4s" repeatCount="indefinite"/></path></svg></div>')
    items = [f'<div class="step"><div class="num">{i}</div><div class="stxt">{t}</div></div>'
             for i, t in enumerate(steps, 1)]
    return '<div class="flow">' + arrow.join(items) + "</div>"


def quality_color(v):
    """Table cell colour: teal for a full rep, orange for a partial rep."""
    return f"color: {TEAL}; font-weight: 600" if v == "full" else f"color: {ORANGE}; font-weight: 600"


st.set_page_config(page_title="RepCounter", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 4rem; max-width: 1250px;}}
.hero {{display: flex; align-items: center; justify-content: space-between; gap: 20px;
       background: linear-gradient(120deg, #E8F7F2 0%, #D6F0E8 60%, #BFE8DB 140%);
       border: 1px solid #CFE6DE; border-radius: 18px; padding: 24px 34px; margin-bottom: 22px;}}
.hero .title {{font-size: 2.7rem; font-weight: 800; letter-spacing: -1px; line-height: 1.1; color: {NAVY};}}
.hero .sub {{margin: 8px 0 16px 0; color: #3F5873; font-size: 1.05rem;}}
.hero svg {{width: 280px; height: 160px;}}
.chip {{display: inline-block; padding: 3px 13px; margin: 0 8px 4px 0; border-radius: 999px;
       background: rgba(15,163,127,.10); border: 1px solid rgba(15,163,127,.30);
       font-size: .8rem; color: #0B6E58;}}
.card {{background: #FFFFFF; border: 1px solid #DCE4EE; border-radius: 14px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 3px rgba(20,33,61,.06);}}
.card .lab {{color: #5B6B82; font-size: .75rem; text-transform: uppercase; letter-spacing: .09em;}}
.card .val {{font-size: 2.1rem; font-weight: 750; line-height: 1.25; color: {NAVY};}}
.card .txt {{color: #51627A; font-size: .92rem; margin-top: 4px;}}
.card.big {{background: linear-gradient(135deg, #DDF4EC, #FFFFFF); border-color: {TEAL};}}
.card.big .val {{font-size: 2.6rem;}}
.card.ex svg {{width: 100%; height: 165px; margin-bottom: 6px;}}
.ptitle {{color: #5B6B82; font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; margin: 6px 0 8px 2px;}}
.badge {{display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600;
        background: rgba(15,163,127,.12); color: #0B7A5F; border: 1px solid rgba(15,163,127,.4);}}
.flow {{display: flex; flex-direction: column; align-items: center;}}
.step {{display: flex; align-items: center; gap: 16px; width: 100%; background: #FFFFFF;
       border: 1px solid #DCE4EE; border-radius: 14px; padding: 14px 20px;
       box-shadow: 0 1px 3px rgba(20,33,61,.06);}}
.step .num {{flex: none; width: 38px; height: 38px; border-radius: 50%; background: {TEAL}; color: white;
            font-weight: 700; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;}}
.step .stxt {{color: {NAVY}; font-size: 1rem;}}
.arrow {{height: 36px; display: flex; align-items: center;}}
video {{border-radius: 12px;}}
[data-testid="stSidebar"] {{border-right: 1px solid #DCE4EE;}}
</style>
<div class="hero">
  <div>
    <div class="title">RepCounter</div>
    <div class="sub">Counts push-ups, squats and bicep curls from a video, and checks every rep.</div>
    <span class="chip">YOLOv8n-pose</span><span class="chip">TensorRT FP16</span>
    <span class="chip">ByteTrack</span><span class="chip">Jetson Orin Nano</span>
  </div>
  {exercise_svg("pushup")}
</div>
""", unsafe_allow_html=True)

# ---------------- sidebar: choose the exercise and the video ----------------
with st.sidebar:
    st.markdown("### Settings")
    label = st.selectbox("Exercise", list(EXERCISE_NAMES))
    exercise = EXERCISE_NAMES[label]
    source = st.radio("Video source", ["Sample clips", "Upload video"])
    clip_path = None
    if source == "Sample clips":
        clips = sorted(p for p in glob.glob("data/clips/*")
                       if os.path.basename(p).startswith(CLIP_PREFIX[exercise]))
        if clips:
            clip_path = st.selectbox("Clip", clips, format_func=os.path.basename)
        else:
            st.info("No sample clips for this exercise.")
    else:
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
        if up is not None:
            os.makedirs(UPLOADS, exist_ok=True)
            clip_path = f"{UPLOADS}/upload{os.path.splitext(up.name)[1]}"
            with open(clip_path, "wb") as fh:
                fh.write(up.getbuffer())
    have_engine = os.path.exists(ENGINE)
    use_trt = st.checkbox("Use TensorRT engine (faster)", value=have_engine, disabled=not have_engine)
    run = st.button("Run counter", type="primary", disabled=clip_path is None)

# ---------------- run the counter ----------------
if run:
    with st.spinner("Counting reps..."):
        proc = run_counter(clip_path, exercise, use_trt)
        if proc.returncode != 0:
            st.session_state.pop("result", None)
            st.error("The counter failed:\n\n" + proc.stderr[-800:])
        else:
            name = os.path.splitext(os.path.basename(clip_path))[0]
            m = re.search(r"FPS=([\d.]+)", proc.stdout)
            st.session_state["result"] = dict(name=name, exercise=exercise, label=label,
                                              fps=float(m.group(1)) if m else None,
                                              video=web_video(name))

# ---------------- show the results ----------------
res = st.session_state.get("result")
if res is None:
    # empty state: the three exercises (animated), then the three steps as a vertical flow
    e = st.columns(3)
    for col, (kind, name, text) in zip(e, [
            ("pushup", "Push-ups", "Elbow angle, filmed from the side."),
            ("squat", "Squats", "Knee angle, side or angled view."),
            ("curl", "Bicep curls", "Elbow angle, arm clearly visible.")]):
        col.markdown(f'<div class="card ex">{exercise_svg(kind)}<div class="lab">{name}</div>'
                     f'<div class="txt">{text}</div></div>', unsafe_allow_html=True)
    st.write("")
    mid = st.columns([1, 1.6, 1])[1]
    mid.markdown(steps_html(["Pick an exercise and a video in the sidebar.",
                             "Press Run counter.",
                             "Review the annotated video, the angle plot and the rep table."]),
                 unsafe_allow_html=True)
else:
    reps = pd.read_csv(f"{OUT}/{res['name']}_reps.csv")
    full = int((reps["quality"] == "full").sum())
    st.markdown(f'### {res["label"]} &nbsp; <span class="badge">{res["name"]}</span>', unsafe_allow_html=True)
    avg = f"{reps['duration_s'].mean():.2f}" if len(reps) else "-"
    fps = f"{res['fps']:.1f}" if res["fps"] else "-"
    c = st.columns([1.3, 1, 1, 1, 1])
    c[0].markdown(card("Total reps", len(reps), TEAL, "big"), unsafe_allow_html=True)
    c[1].markdown(card("Full reps", full, TEAL), unsafe_allow_html=True)
    c[2].markdown(card("Partial reps", len(reps) - full, ORANGE), unsafe_allow_html=True)
    c[3].markdown(card("Avg sec / rep", avg), unsafe_allow_html=True)
    c[4].markdown(card("Speed (FPS)", fps), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ptitle">Annotated video</div>', unsafe_allow_html=True)
        st.video(res["video"])
    with right:
        st.markdown('<div class="ptitle">Joint angle over time</div>', unsafe_allow_html=True)
        st.pyplot(angle_plot(res["name"], res["exercise"], reps))

    st.markdown('<div class="ptitle">Reps</div>', unsafe_allow_html=True)
    if len(reps):
        styled = reps.style.format({"start_s": "{:.2f}", "end_s": "{:.2f}", "duration_s": "{:.2f}"})
        styled = (getattr(styled, "map", None) or styled.applymap)(quality_color, subset=["quality"])
        st.dataframe(styled, hide_index=True)
        st.download_button("Download reps CSV", reps.to_csv(index=False),
                           file_name=f"{res['name']}_reps.csv", mime="text/csv")
    else:
        st.warning("No reps were counted in this video.")
    with st.expander("How reps are counted"):
        st.write("A rep starts when the joint angle drops below the red line and ends when it rises "
                 "above the blue line. A rep is **full** if the lowest angle reached the target depth, "
                 "otherwise it is **partial**. Two different lines stop small jitter from counting twice.")
