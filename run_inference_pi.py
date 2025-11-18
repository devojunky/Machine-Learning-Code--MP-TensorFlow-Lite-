"""
Step 3 of 3 Run gesture recognition inference (Raspberry Pi - XGBoost only)
Example usage: python run_inference_pi.py --models models [--headless]

This version is optimized for Raspberry Pi running Python 3.11 with XGBoost only (no TensorFlow).
It uses Linux/BlueZ media controls for Bluetooth audio devices.

------ Gestures ------
Track control (index finger down):
  Next track:      point thumb right, palm closed
  Previous track:  point thumb left, palm closed
Volume control (index finger up):
  Volume up:       twirl index fingers clockwise
  Volume down:     twirl index fingers counter-clockwise
Play / Pause (either mode):
  Shakaaaaaa 🤙🤙🤙

Usage:
  python run_inference_pi.py --models models [--headless]
Options:
  --headless   Run without GUI window (useful on Raspberry Pi)
"""

import cv2, time, math, json, argparse, os, threading, queue, sys, subprocess
import numpy as np
import mediapipe as mp
import joblib

# ---------------- Picamera2 Capture (replaces GStreamer) ----------------
class PiCamera2Capture:
    def __init__(self, W=640, H=360):
        self.W, self.H = W, H
        self.picam2 = None
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (self.W, self.H)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("[INFO] Picamera2 backend initialized successfully.")
        except ImportError:
            print("[WARN] Picamera2 library not found. Use 'pip install picamera2'.")
            self.picam2 = None
        except Exception as e:
            print(f"[WARN] Failed to initialize Picamera2: {e}")
            self.picam2 = None

    def read(self):
        if not self.isOpened():
            return False, None
        # Convert RGB frame from Picamera2 to BGR for OpenCV
        frame = self.picam2.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        if self.picam2:
            self.picam2.stop()

    def isOpened(self):
        return self.picam2 is not None and self.picam2.started


# ---------------- Non-blocking command worker ----------------
class CommandWorker(threading.Thread):
    def __init__(self, max_queue=64):
        super().__init__(daemon=True); self.q = queue.Queue(maxsize=max_queue)
    def run(self):
        while True:
            fn, args = self.q.get()
            try: fn(*args)
            except Exception as e: print("[cmd worker]", e, file=sys.stderr)
            finally: self.q.task_done()
    def submit(self, fn, *args):
        try: self.q.put_nowait((fn, args))
        except queue.Full: pass

worker = CommandWorker(); worker.start()

# ---------------- Linux/BlueZ media controls (replaces macOS AppleScript) ----------------
def _btctl(cmds):
    """
    Send a small sequence of commands to bluetoothctl. Example cmds:
      ["menu player", "play-pause", "back"]
    Requires an active phone connection with a MediaPlayer.
    """
    try:
        subprocess.run(
            ["bluetoothctl"],
            input=("\n".join(cmds) + "\n").encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
    except Exception as e:
        print("[bluetoothctl]", e, file=sys.stderr)

def _bt_playpause(): _btctl(["menu player", "play-pause", "back"])
def _bt_next():      _btctl(["menu player", "next", "back"])
def _bt_prev():      _btctl(["menu player", "previous", "back"])
def _bt_vol_up():    _btctl(["menu player", "volumeup", "back"])
def _bt_vol_down():  _btctl(["menu player", "volumedown", "back"])

def _alsa_nudge(db_delta):
    """
    Local volume change if AVRCP absolute volume is unsupported.
    Adjust 'Master' to your mixer name if needed (e.g., 'PCM' or specific sink).
    """
    try:
        subprocess.run(["amixer", "set", "Master", f"{int(db_delta)}dB"], check=False)
    except Exception as e:
        print("[amixer]", e, file=sys.stderr)

def async_next_track():        worker.submit(_bt_next)
def async_prev_track():        worker.submit(_bt_prev)
def async_toggle_play_pause(): worker.submit(_bt_playpause)

def async_nudge_vol(delta):
    """
    Keep the original API (delta in arbitrary units). We map to AVRCP
    up/down steps; uncomment ALSA fallback if phone volume doesn’t move.
    """
    if delta > 0:
        worker.submit(_bt_vol_up)
        # worker.submit(_alsa_nudge, +2)   # optional fallback
    else:
        worker.submit(_bt_vol_down)
        # worker.submit(_alsa_nudge, -2)   # optional fallback

# ---------------- MediaPipe & features (match training) ----------------
SEL = [0,1,2,3,4,5,9,13,17,6,8,10,12,14,16,18,20]
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# --- Tunables for runtime thresholds ---
PP_THRESH = 0.55    # play/pause probability threshold
PP_HOLD   = 2       # require N consecutive frames above threshold

def _rot2d(pt, cs, sn):
    x,y = pt
    return np.array([cs*x - sn*y, sn*x + cs*y], dtype=np.float32)

class WinBuf:
    def __init__(self, W): self.W=W; self.buf=[]
    def add(self, x): self.buf.append(x); self.buf=self.buf[-self.W:]
    def ready(self):  return len(self.buf)==self.W
    def stack(self):  return np.stack(self.buf)

def extract_features(lm, handed_label, prev_state):
    P = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(21)], dtype=np.float32)
    P -= P[0]
    palm_w = np.linalg.norm(P[5,:2] - P[17,:2]) + 1e-6
    P /= palm_w
    v = P[5,:2] - P[17,:2]
    L = np.linalg.norm(v) + 1e-9
    cs, sn = v[0]/L, v[1]/L
    C = np.zeros((21,2), dtype=np.float32)
    for i in range(21):
        C[i] = _rot2d(P[i,:2], cs, sn)

    coords = C[SEL].reshape(-1)
    curls = np.array([C[4,1]-C[3,1], C[8,1]-C[6,1], C[12,1]-C[10,1], C[16,1]-C[14,1], C[20,1]-C[18,1]], dtype=np.float32)
    td = C[4]-C[3]; td = td/(np.linalg.norm(td)+1e-9)
    nz = np.cross(P[5], P[17])[2:3]
    if prev_state is None:
        d_idx = np.zeros(2, np.float32); d_td=np.zeros(2, np.float32)
    else:
        d_idx = C[8]-prev_state['idx']; d_td = td-prev_state['td']
    hflag = np.array([1.0 if handed_label=='Right' else 0.0], dtype=np.float32)
    feat = np.concatenate([coords, curls, td, nz, d_idx, d_td, hflag]).astype(np.float32)
    return feat, {'idx':C[8].copy(), 'td':td.copy()}, C

# --------- XGB backend (existing) ----------
class XGBBackend:
    def __init__(self, models_dir, window):
        self.tracks = joblib.load(os.path.join(models_dir, 'tracks_xgb.joblib'))
        self.volcls = joblib.load(os.path.join(models_dir, 'vol_cls_xgb.joblib'))
        try:
            self.play = joblib.load(os.path.join(models_dir, 'play_xgb.joblib'))
        except Exception:
            self.play = None
        self.W = window

    def make_window_features(self, win_arr):
        # [mean(F), std(F), dxy(index_tip), ang_sum_deg]; must match training
        X = win_arr
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        INDEX_IN_SEL = SEL.index(8)
        idx_off = 2*INDEX_IN_SEL
        dxy = X[-1, idx_off:idx_off+2] - X[0, idx_off:idx_off+2]
        # angle sum from index tip around centroid
        series = X[:, idx_off:idx_off+2]
        c = series.mean(axis=0)
        vecs = series - c
        norms = (np.linalg.norm(vecs, axis=1)+1e-9)[:,None]
        v = vecs / norms
        ang = 0.0
        for t in range(1, v.shape[0]):
            ax, ay = v[t-1]; bx, by = v[t]
            da = math.atan2(by,bx) - math.atan2(ay,ax)
            if da > math.pi: da -= 2*math.pi
            if da < -math.pi: da += 2*math.pi
            ang += -da
        ang_deg = math.degrees(ang)
        return np.concatenate([mean, std, dxy, [ang_deg]]).astype(np.float32)

    def predict(self, win_arr):
        x = self.make_window_features(win_arr)[None, :]
        p_tr = self.tracks.predict_proba(x)[0]  # len 3
        p_vl = self.volcls.predict_proba(x)[0]  # len 3
        p_pp = None
        if self.play is not None:
            p_pp = self.play.predict_proba(x)[0][1]  # scalar prob
        return p_tr, p_vl, p_pp

# ---------------- Main loop ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--headless", action="store_true", help="Run without GUI window (Pi-friendly)")
    args = ap.parse_args()

    # Load meta for window size
    W_meta = None
    meta_path = os.path.join(args.models, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f: W_meta = json.load(f).get("window")
    
    # If no meta, default to 10 (XGBoost training default)
    if W_meta is None:
        W_meta = 10
        print(f"[WARNING] No meta.json found, using default window size: {W_meta}")

    # Load XGBoost backend only
    xgb = XGBBackend(args.models, window=W_meta)

    # Camera + buffers (Pi-friendly: try Picamera2 first, then fallback to /dev/video0)
    print("[INFO] Attempting to open camera with Picamera2...")
    cap = PiCamera2Capture(W=640, H=360)
    
    if not cap.isOpened():
        print("[WARN] Picamera2 failed. Falling back to default camera index 0...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[INFO] Successfully opened camera at index 0.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("\n[ERROR] Could not open any camera. Please check your configuration.")
        print("    1. Is the camera enabled? Run 'sudo raspi-config' -> Interface Options -> CSI Camera -> Enable.")
        print("    2. Is the camera detected? Run 'libcamera-hello --list-cameras' or 'picamera2-hello' in your terminal.")
        print("    3. Is the ribbon cable connected correctly at both ends?")
        exit()

    print("[INFO] Camera opened successfully. Starting gesture detection...")

    prev_state=None
    win = WinBuf(W_meta or 10)
    cooldown_track = cooldown_vol = cooldown_pause = 0.0
    hold_next = hold_prev = 0
    hold_pp = 0
    LOCK_AFTER_PAUSE = 0.4

    # Palm-gated index_up (prevents back-of-hand false volume)
    NZ_THRESH = 0.02
    PALM_FRONT_SIGN = -1  # flip to +1 if your camera orientation disagrees
    COORDS_LEN = 34; CURLS_LEN=5; TD_LEN=2
    NZ_POS = COORDS_LEN + CURLS_LEN + TD_LEN

    with mp_hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=0,
                        min_detection_confidence=0.5,min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            now = time.time()
            info = ""

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label
                if not args.headless:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(), mp_style.get_default_hand_connections_style())
                feat, prev_state, C = extract_features(hand.landmark, handed, prev_state)
                win.add(feat)

                # Mode gate with palm orientation
                nz = float(feat[NZ_POS])
                palm_front = (PALM_FRONT_SIGN * nz) > NZ_THRESH
                index_up = (C[8,1] < C[6,1]) and palm_front

                if win.ready():
                    win_arr = win.stack()  # [W, Fraw]

                    # XGBoost predictions only
                    p_tr, p_vl, p_pp = xgb.predict(win_arr)
                    if xgb.play is None:
                        p_pp = None

                    # --- Play/Pause (shaka) first ---
                    if p_pp is not None:
                        hold_pp = hold_pp + 1 if p_pp > PP_THRESH else 0
                    if (p_pp is not None) and (hold_pp >= PP_HOLD) and now > cooldown_pause:
                        async_toggle_play_pause()
                        info = '⏯️ Play/Pause'
                        cooldown_pause = now + 0.9
                        cooldown_track = now + LOCK_AFTER_PAUSE
                        cooldown_vol   = now + LOCK_AFTER_PAUSE
                        hold_pp = 0
                    else:
                        # Volume vs tracks by mode
                        if index_up and now > cooldown_vol:
                            if p_vl[2] > 0.65:
                                async_nudge_vol(+5); info = '🔊 Vol +'; cooldown_vol = now + 0.12
                            elif p_vl[0] > 0.65:
                                async_nudge_vol(-5); info = '🔉 Vol −'; cooldown_vol = now + 0.12
                        elif now > cooldown_track:
                            hold_next = hold_next + 1 if p_tr[1] > 0.65 else 0
                            hold_prev = hold_prev + 1 if p_tr[2] > 0.65 else 0
                            if hold_next >= 3:
                                async_next_track(); info = '⏭️ Next'; cooldown_track = now + 0.35; hold_next=hold_prev=0
                            elif hold_prev >= 3:
                                async_prev_track(); info = '⏮️ Previous'; cooldown_track = now + 0.35; hold_next=hold_prev=0

                mode = 'VOLUME (index up)' if index_up else 'TRACKS (index down)'
                if not args.headless:
                    cv2.putText(frame, mode, (10,24), 0, 0.7, (255,255,255), 2)
            else:
                prev_state=None; win = WinBuf(W_meta or 10)

            # Overlay debug: show play/pause probability if available
            try:
                if p_pp is not None and (not args.headless):
                    cv2.putText(frame, f"PP: {p_pp:.2f}", (10, 76), 0, 0.6, (0,200,255), 2)
            except Exception:
                pass

            if not args.headless:
                if info:
                    cv2.putText(frame, info, (10, 50), 0, 0.8, (0,255,0), 2)
                cv2.putText(frame, "q = quit", (10, h-12), 0, 0.6, (200,200,200), 2)

                cv2.imshow('Gesture — Inference (XGBoost)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                # tiny sleep to keep CPU sane when headless
                time.sleep(0.001)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
