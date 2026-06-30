"""
╔══════════════════════════════════════════════════════════════╗
║          YOLO FITNESS TRACKER  —  Multi-Person HUD           ║
║  Reps · Sets · Form · Calories · Pace · MP4 Export           ║
╚══════════════════════════════════════════════════════════════╝
Controls:
  Q  — Quit
  R  — Reset all trackers
  E  — Cycle exercise  (squat → deadlift → curl → shoulder_press)
  S  — Save snapshot
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# ──────────────────────────────────────────────────────────────
# CONFIG  ← edit here
# ──────────────────────────────────────────────────────────────
SOURCE        = "videos/vid5.mp4"                       # 0 = webcam | "videos/vid5.mp4"
YOLO_MODEL    = "yolo26n-pose.pt"       # n / s / m / l
FRAME_W       = 720
FRAME_H       = 1280
SAVE_OUTPUT   = True
OUTPUT_PATH   = "workout_output.mp4"
REPS_PER_SET  = 10                      # reps that make 1 set
MAX_PERSONS   = 2                       # track up to 2 people
EXERCISE      = "deadlift"                  # squat | deadlift | curl | shoulder_press
MET_VALUE     = 5.0                     # metabolic equivalent (adjust per exercise)
BODY_WEIGHT   = 70                      # kg — for calorie estimate

# ──────────────────────────────────────────────────────────────
# EXERCISE DEFINITIONS
#   each entry: (joint_a, joint_b, joint_c, down_thresh, up_thresh, label)
#   YOLO keypoint indices:
#   0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
#   5=l_sho 6=r_sho 7=l_elb 8=r_elb 9=l_wri 10=r_wri
#   11=l_hip 12=r_hip 13=l_kne 14=r_kne 15=l_ank 16=r_ank
# ──────────────────────────────────────────────────────────────
EXERCISES = {
    "squat": {
        "label"      : "SQUATS",
        "joints"     : [(11,13,15), (12,14,16)],   # L-knee, R-knee
        "down_thresh": 100,
        "up_thresh"  : 160,
        "good_lo"    : 80,
        "good_hi"    : 170,
        "primary"    : "knee",
    },
    "deadlift": {
        "label"      : "DEADLIFT",
        "joints"     : [(5,11,13), (6,12,14)],     # L-hip, R-hip
        "down_thresh": 80,
        "up_thresh"  : 155,
        "good_lo"    : 70,
        "good_hi"    : 180,
        "primary"    : "hip",
    },
    "curl": {
        "label"      : "BICEP CURL",
        "joints"     : [(5,7,9), (6,8,10)],         # L-elbow, R-elbow
        "down_thresh": 50,
        "up_thresh"  : 140,
        "good_lo"    : 30,
        "good_hi"    : 160,
        "primary"    : "elbow",
    },
    "shoulder_press": {
        "label"      : "SHOULDER PRESS",
        "joints"     : [(5,7,9), (6,8,10)],
        "down_thresh": 80,
        "up_thresh"  : 155,
        "good_lo"    : 70,
        "good_hi"    : 180,
        "primary"    : "elbow",
    },
}

EXERCISE_CYCLE = list(EXERCISES.keys())

# ──────────────────────────────────────────────────────────────
# YOLO skeleton connections
# ──────────────────────────────────────────────────────────────
YOLO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# ──────────────────────────────────────────────────────────────
# UI PALETTE  — dark-tech / neon-teal theme
# ──────────────────────────────────────────────────────────────
C_BG         = (14, 14, 18)
C_PANEL      = (22, 24, 32)
C_ACCENT     = (0, 220, 180)         # neon teal
C_ACCENT2    = (255, 180, 0)         # amber
C_GOOD       = (50, 220, 100)        # green
C_WARN       = (0, 200, 255)         # cyan
C_BAD        = (60, 60, 240)         # red-ish (BGR)
C_WHITE      = (240, 240, 240)
C_GREY       = (120, 120, 130)
C_DARK       = (35, 38, 50)

# Per-person accent colors
PERSON_COLORS = [(0, 220, 180), (255, 160, 50)]   # teal / amber

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    a  = np.array(a, dtype=float)
    b  = np.array(b, dtype=float)
    c  = np.array(c, dtype=float)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 1)

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.72, radius=10):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+radius,y1), (x2-radius,y2), color, -1)
    cv2.rectangle(overlay, (x1,y1+radius), (x2,y2-radius), color, -1)
    for cx,cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                  (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(overlay,(cx,cy),radius,color,-1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def text(img, msg, x, y, size=0.5, color=C_WHITE, bold=1, aa=True):
    cv2.putText(img, str(msg), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, bold, cv2.LINE_AA if aa else 0)

def text_c(img, msg, cx, y, size=0.5, color=C_WHITE, bold=1):
    """Horizontally centered text."""
    w, _ = cv2.getTextSize(str(msg), cv2.FONT_HERSHEY_SIMPLEX, size, bold)[0]
    text(img, msg, cx - w//2, y, size, color, bold)

def bar_v(img, x, y, w, h, pct, fg_color, bg=(40,42,55), border=True):
    """Vertical fill bar (bottom to top)."""
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    if border:
        cv2.rectangle(img, (x,y), (x+w, y+h), C_GREY, 1)
    fill = int(pct * h)
    if fill > 0:
        cv2.rectangle(img, (x, y+h-fill), (x+w, y+h), fg_color, -1)

def bar_h(img, x, y, w, h, pct, fg_color, bg=(40,42,55)):
    """Horizontal fill bar."""
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    fill = int(pct * w)
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill, y+h), fg_color, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), C_GREY, 1)

# ──────────────────────────────────────────────────────────────
# Per-person Tracker
# ──────────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self, pid, color):
        self.pid          = pid
        self.color        = color
        self.reps         = 0
        self.sets         = 0
        self.stage        = None
        self.rep_times    = []
        self.last_rep_t   = time.time()
        self.angle_hist   = deque(maxlen=90)
        self.smooth_buf   = {}
        self.form_score   = 1.0           # 0–1
        self.calories     = 0.0
        self.flash        = 0             # frames to flash rep counter
        self.last_angles  = {}            # joint_key → angle

    def smooth(self, key, val, win=6):
        if key not in self.smooth_buf:
            self.smooth_buf[key] = deque(maxlen=win)
        self.smooth_buf[key].append(val)
        return float(np.mean(self.smooth_buf[key]))

    def update(self, primary_angle, ex_cfg, dt_sec):
        self.angle_hist.append(primary_angle)

        ex = ex_cfg
        lo, hi = ex["down_thresh"], ex["up_thresh"]
        gl, gh = ex["good_lo"], ex["good_hi"]

        # form score: how centered in good range
        if gl <= primary_angle <= gh:
            self.form_score = min(1.0, self.form_score + 0.05)
        else:
            self.form_score = max(0.0, self.form_score - 0.08)

        # stage machine
        if primary_angle < lo:
            self.stage = "down"
        if primary_angle > hi and self.stage == "down":
            self.stage  = "up"
            self.reps  += 1
            if self.reps % REPS_PER_SET == 0:
                self.sets += 1
            now = time.time()
            self.rep_times.append(now - self.last_rep_t)
            self.last_rep_t = now
            self.flash = 20

        # calories  (MET × weight × time_hours)
        self.calories += MET_VALUE * BODY_WEIGHT * (dt_sec / 3600)

    @property
    def avg_pace(self):
        if len(self.rep_times) < 2:
            return 0.0
        return float(np.mean(self.rep_times[-5:]))

    def form_label(self):
        s = self.form_score
        if s > 0.75: return "GOOD FORM",  C_GOOD
        if s > 0.45: return "WATCH FORM", C_WARN
        return "FIX FORM", C_BAD

# ──────────────────────────────────────────────────────────────
# Panel drawing
# ──────────────────────────────────────────────────────────────
PANEL_W  = 190
PANEL_H  = 380
BAR_W    = 30
BAR_H    = 200

def draw_person_panel(img, tracker: PersonTracker, ex_cfg, side="left"):
    """Draw the side analytics panel for one person."""
    if side == "left":
        px = 12
    else:
        px = FRAME_W - PANEL_W - BAR_W - 24

    py = 12
    pc = tracker.color

    # Panel background
    alpha_rect(img, px, py, px+PANEL_W, py+PANEL_H, C_PANEL, alpha=0.80, radius=12)

    # Person ID badge
    badge_x, badge_y = px+10, py+10
    cv2.rectangle(img, (badge_x,badge_y), (badge_x+44,badge_y+24), pc, -1)
    text(img, f"P{tracker.pid}", badge_x+8, badge_y+18, 0.55, C_BG, 2)

    # Confidence %
    conf_pct = int(tracker.form_score * 100)
    text(img, f"{conf_pct}%", px+PANEL_W-55, py+26, 0.6, pc, 2)

    yc = py + 44

    # Exercise label
    text(img, ex_cfg["label"], px+10, yc, 0.42, C_GREY, 1)
    yc += 22

    # REPS — big
    rep_color = C_GOOD if tracker.flash > 0 else C_WHITE
    if tracker.flash > 0: tracker.flash -= 1
    text(img, str(tracker.reps), px+10, yc+46, 2.4, rep_color, 4)
    yc += 62

    # SETS label + number
    text(img, "SETS", px+10, yc, 0.42, C_GREY, 1)
    text(img, str(tracker.sets), px+10, yc+32, 1.5, C_ACCENT2, 3)
    yc += 52

    # Stage badge
    stage_txt = (tracker.stage or "---").upper()
    s_col     = C_GOOD if tracker.stage == "up" else (C_ACCENT if tracker.stage == "down" else C_GREY)
    cv2.rectangle(img, (px+6, yc), (px+PANEL_W-6, yc+28), s_col, -1)
    text_c(img, stage_txt, px+PANEL_W//2, yc+20, 0.62, C_BG, 2)
    yc += 38

    # Form label badge
    f_label, f_col = tracker.form_label()
    cv2.rectangle(img, (px+6, yc), (px+PANEL_W-6, yc+24), C_DARK, -1)
    cv2.rectangle(img, (px+6, yc), (px+PANEL_W-6, yc+24), f_col, 1)
    text_c(img, f_label, px+PANEL_W//2, yc+17, 0.45, f_col, 1)
    yc += 34

    # Calories
    text(img, f"~{tracker.calories:.1f} kcal", px+10, yc, 0.42, C_GREY, 1)
    yc += 20

    # Angle history sparkline
    hist = list(tracker.angle_hist)
    if len(hist) > 4:
        sh = 38
        sw = PANEL_W - 18
        sx, sy = px+8, yc
        cv2.rectangle(img, (sx,sy), (sx+sw, sy+sh), (28,30,42), -1)
        pts = [(sx+int(i/(len(hist)-1)*sw),
                sy+sh-int(np.clip(v/180,0,1)*sh))
               for i,v in enumerate(hist)]
        for i in range(1,len(pts)):
            cv2.line(img, pts[i-1], pts[i], pc, 1, cv2.LINE_AA)
        # threshold lines
        for thr, tc in [(ex_cfg["down_thresh"], C_BAD),
                        (ex_cfg["up_thresh"],   C_GOOD)]:
            ty = sy+sh - int(np.clip(thr/180,0,1)*sh)
            cv2.line(img,(sx,ty),(sx+sw,ty),tc,1)

    # Vertical form/depth bar (to the right of panel)
    bx = px + PANEL_W + 8
    by = py + 40
    bar_v(img, bx, by, BAR_W, BAR_H, tracker.form_score, C_GOOD)
    text_c(img, "FORM", bx+BAR_W//2, by+BAR_H+14, 0.35, C_GREY, 1)

    # Angle values block (bottom of bar)
    ay = by + BAR_H + 24
    for jk, jv in list(tracker.last_angles.items())[:3]:
        text(img, f"{jk}: {jv:.0f}°", bx-10, ay, 0.35, C_GREY, 1)
        ay += 14


def draw_center_banner(img, n_persons, ex_cfg, trackers, elapsed):
    """Top-center banner: exercise name + total reps."""
    bw, bh = 360, 44
    bx = FRAME_W//2 - bw//2
    alpha_rect(img, bx, 8, bx+bw, 8+bh, C_PANEL, alpha=0.85, radius=10)
    total = sum(t.reps for t in trackers[:n_persons])
    ex_name = ex_cfg["label"]
    text_c(img, f"TOTAL {ex_name}: {total}", FRAME_W//2, 36, 0.7, C_ACCENT, 2)


def draw_bottom_bar(img, trackers, n_persons, elapsed):
    """Bottom info strip."""
    bh = 34
    cv2.rectangle(img,(0,FRAME_H-bh),(FRAME_W,FRAME_H),(12,13,18),-1)
    cv2.line(img,(0,FRAME_H-bh),(FRAME_W,FRAME_H-bh),C_ACCENT,1)

    m, s  = divmod(elapsed, 60)
    items = [f"TIME  {m:02d}:{s:02d}"]
    for i, t in enumerate(trackers[:n_persons]):
        items.append(f"P{t.pid} avg {t.avg_pace:.1f}s")
    for i, t in enumerate(trackers[:n_persons]):
        items.append(f"P{t.pid}: {t.reps} reps / {t.sets} sets")

    seg_w = FRAME_W // len(items)
    for idx, item in enumerate(items):
        col = PERSON_COLORS[idx-1] if idx > 0 and idx <= n_persons else C_WHITE
        cx  = idx * seg_w + seg_w//2
        if idx > 0:
            cv2.line(img,(idx*seg_w, FRAME_H-bh+4),(idx*seg_w, FRAME_H-4),C_GREY,1)
        text_c(img, item, cx, FRAME_H-10, 0.42, col, 1)


def draw_joint_angle(img, px, py, angle, color, radius=22):
    """Draw mini arc + value at a joint."""
    sweep = int(np.clip(angle, 0, 180))
    for d in range(0, sweep, 4):
        rad = np.radians(d - 90)
        ex_ = int(px + radius * np.cos(rad))
        ey_ = int(py + radius * np.sin(rad))
        cv2.circle(img,(ex_,ey_),2,color,-1,cv2.LINE_AA)

    # Dark badge behind number
    label = f"{angle:.0f}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    bx1, by1  = px-tw//2-6, py+radius-4
    bx2, by2  = px+tw//2+6, py+radius+th+4
    alpha_rect(img, bx1, by1, bx2, by2, C_BG, alpha=0.7, radius=4)
    text_c(img, label, px, by2-4, 0.5, color, 2)


def draw_skeleton(img, kps, color, confs=None, thresh=0.4):
    """Draw YOLO skeleton with glowing effect."""
    for (i,j) in YOLO_SKELETON:
        if i>=len(kps) or j>=len(kps): continue
        x1,y1 = int(kps[i][0]), int(kps[i][1])
        x2,y2 = int(kps[j][0]), int(kps[j][1])
        if (x1==0 and y1==0) or (x2==0 and y2==0): continue
        # glow: thick dim line + thin bright line
        cv2.line(img,(x1,y1),(x2,y2),(color[0]//3,color[1]//3,color[2]//3),5,cv2.LINE_AA)
        cv2.line(img,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)

    for ki,(x,y) in enumerate(kps):
        x,y = int(x),int(y)
        if x==0 and y==0: continue
        if confs is not None and confs[ki]<thresh: continue
        cv2.circle(img,(x,y),7,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(img,(x,y),5,color,-1,cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# Person-label overlay (name badge above head)
# ──────────────────────────────────────────────────────────────
def draw_person_label(img, kps, pid, color):
    nose = kps[0]
    if nose[0]==0 and nose[1]==0:
        return
    hx, hy = int(nose[0]), int(nose[1]) - 40
    label = f"P{pid}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img,(hx-tw//2-8,hy-th-6),(hx+tw//2+8,hy+6), color, -1)
    text_c(img, label, hx, hy, 0.65, C_BG, 2)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
yolo_model    = YOLO(YOLO_MODEL)
ex_idx        = EXERCISE_CYCLE.index(EXERCISE)
trackers      = [PersonTracker(i+1, PERSON_COLORS[i]) for i in range(MAX_PERSONS)]
start_time    = time.time()
prev_time     = start_time
frame_count   = 0
snapshot_count= 0

cap       = cv2.VideoCapture(SOURCE)
is_webcam = SOURCE == 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (FRAME_W, FRAME_H))
    print(f"📹  Recording → {OUTPUT_PATH}")

print(f"▶   Source  : {'Webcam' if is_webcam else SOURCE}")
print(f"▶   Model   : {YOLO_MODEL}")
print(f"▶   Exercise: {EXERCISE}  (press E to cycle)")
print(f"▶   Q=Quit  R=Reset  E=Exercise  S=Snapshot")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if not is_webcam:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        break

    if is_webcam:
        frame = cv2.flip(frame, 1)

    frame_count += 1
    now    = time.time()
    dt     = now - prev_time
    prev_time = now
    elapsed= int(now - start_time)

    image  = cv2.resize(frame, (FRAME_W, FRAME_H))

    # ── YOLO inference ─────────────────────────────────────────
    yolo_res = yolo_model.predict(image, conf=0.45, verbose=False)
    ex_cfg   = EXERCISES[EXERCISE_CYCLE[ex_idx]]

    n_detected = 0
    if yolo_res and yolo_res[0].keypoints is not None:
        all_kps   = yolo_res[0].keypoints.xy.cpu().numpy()
        all_confs = (yolo_res[0].keypoints.conf.cpu().numpy()
                     if yolo_res[0].keypoints.conf is not None else None)
        n_detected = min(len(all_kps), MAX_PERSONS)

        for pidx in range(n_detected):
            kps    = all_kps[pidx]
            confs  = all_confs[pidx] if all_confs is not None else None
            t      = trackers[pidx]
            color  = t.color

            # Draw skeleton
            draw_skeleton(image, kps, color, confs)
            draw_person_label(image, kps, t.pid, color)

            # Calculate angles for this exercise
            angles_raw = {}
            for joint_idx, (ai, bi, ci) in enumerate(ex_cfg["joints"]):
                if ai>=len(kps) or bi>=len(kps) or ci>=len(kps): continue
                ax,ay = kps[ai]; bx,by = kps[bi]; cx,cy = kps[ci]
                if (ax==0 and ay==0) or (bx==0 and by==0) or (cx==0 and cy==0): continue
                raw_ang = calc_angle([ax,ay],[bx,by],[cx,cy])
                s_key   = f"j{joint_idx}"
                ang     = t.smooth(s_key, raw_ang)
                jlabel  = ["L","R"][joint_idx % 2]+"."+ex_cfg["primary"].upper()
                angles_raw[jlabel] = ang

                # pick arc color
                gl,gh = ex_cfg["good_lo"], ex_cfg["good_hi"]
                arc_col = (C_GOOD if gl<=ang<=gh
                           else C_WARN if abs(ang-gl)<20 or abs(ang-gh)<20
                           else C_BAD)
                draw_joint_angle(image, int(bx), int(by), ang, arc_col)

            t.last_angles = angles_raw

            # Rep counting — use average of all joint angles
            if angles_raw:
                avg_ang = float(np.mean(list(angles_raw.values())))
                t.update(avg_ang, ex_cfg, dt)

    # ── Draw panels ────────────────────────────────────────────
    draw_center_banner(image, n_detected, ex_cfg, trackers, elapsed)

    for i in range(min(n_detected, MAX_PERSONS)):
        side = "left" if i == 0 else "right"
        draw_person_panel(image, trackers[i], ex_cfg, side)

    draw_bottom_bar(image, trackers, n_detected, elapsed)

    # ── Top-right: mode / rec indicator ───────────────────────
    alpha_rect(image, FRAME_W-160, 8, FRAME_W-8, 44, C_PANEL, 0.75, 8)
    if SAVE_OUTPUT:
        # Blinking red dot
        if (frame_count // 15) % 2 == 0:
            cv2.circle(image,(FRAME_W-22,26),7,(0,0,220),-1)
        text(image,"REC", FRAME_W-60, 32, 0.5, (0,80,220), 1)
    text(image, EXERCISE_CYCLE[ex_idx].upper(), FRAME_W-155, 32, 0.4, C_GREY, 1)

    # ── Save & show ────────────────────────────────────────────
    if writer:
        writer.write(image)

    cv2.imshow("Fitness Tracker — YOLO HUD", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        trackers   = [PersonTracker(i+1, PERSON_COLORS[i]) for i in range(MAX_PERSONS)]
        start_time = time.time()
        print("✔  Trackers reset")
    elif key == ord('e'):
        ex_idx = (ex_idx + 1) % len(EXERCISE_CYCLE)
        print(f"✔  Exercise → {EXERCISE_CYCLE[ex_idx]}")
    elif key == ord('s'):
        snap = f"snapshot_{snapshot_count:03d}.jpg"
        cv2.imwrite(snap, image)
        snapshot_count += 1
        print(f"📸  Saved {snap}")

# ── Cleanup ────────────────────────────────────────────────────
cap.release()
if writer:
    writer.release()
    print(f"✅  Video saved → {OUTPUT_PATH}")
cv2.destroyAllWindows()

print("\n── WORKOUT SUMMARY ─────────────────────────")
for t in trackers:
    if t.reps > 0:
        print(f"  P{t.pid}  Reps:{t.reps}  Sets:{t.sets}"
              f"  Pace:{t.avg_pace:.1f}s  Calories:{t.calories:.1f} kcal")
print(f"  Duration : {elapsed//60:02d}:{elapsed%60:02d}")
print("─────────────────────────────────────────────")
