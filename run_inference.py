"""
Step 3 of 3 Run gesture recognition inference
Example usage: python run_inference.py --models models
This works for macOS only since it uses apple script to control Music/Spotify only
It's multithreaded so it doesn't mess with inference bc idk why but subprocess calls can be slow and freeze the UI
prolly hard coded limitation from apple script or something
I tested this on macOS 26 and Apple M1 Max. My system was taking up 35 watts during inference with webcam
But that was running at 960x540 and also running spotify and streaming at the same time we could optimize it fs
I'm praying this runs on a Raspberry Pi but idk if inference will do well on rasp
I hope it runs tho 🙏🙏🙏🙏
I also tried hardcoding the gestures and it really didn't work well. Lots of false positives.
ML is wayyyy better
In the future we could add self correction or feedback so you can teach it your own gestures, but that's out of scope for now
Daniel Liang signing out ✌️

------ Gestures ------
Track control (index finger down):
  Next track:      poing thumb right, palm closed
  Previous track:  point thumb left, palm closed
Volume control (index finger up):
  Volume up:       twirl index fingers clockwise
  Volume down:     twirl index fingers counter-clockwise
Play / Pause (either mode):
  Shakaaaaaa 🤙🤙🤙

Run real-time inference with either XGBoost, TFLite, or both (shadow).

Usage:
  python run_inference.py --models models --backend tflite
Options:
  --backend {xgb,tflite,shadow}
"""

import cv2, time, math, json, argparse, os, threading, queue, sys, subprocess
import numpy as np
import mediapipe as mp
import joblib

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

def osa(script):
    subprocess.run(["osascript","-e",script], check=False, capture_output=True)

def _osa_next():
    osa('''
    if application "Spotify" is running then
        tell application "Spotify" to next track
    else if application "Music" is running then
        tell application "Music" to next track
    end if
    ''')

def _osa_prev():
    osa('''
    if application "Spotify" is running then
        tell application "Spotify" to previous track
    else if application "Music" is running then
        tell application "Music" to previous track
    end if
    ''')

def _osa_nudge(delta):
    osa(f'''
    set cur to output volume of (get volume settings)
    set volume output volume (cur + {int(delta)})
    ''')

def _osa_toggle_playpause():
    osa('''
    if application "Spotify" is running then
        tell application "Spotify" to playpause
    else if application "Music" is running then
        tell application "Music" to playpause
    end if
    ''')

def async_next_track(): worker.submit(_osa_next)
def async_prev_track(): worker.submit(_osa_prev)
def async_nudge_vol(delta): worker.submit(_osa_nudge, delta)
def async_toggle_play_pause(): worker.submit(_osa_toggle_playpause)

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

# --------- TFLite backend (new) ----------
class TFLiteBackend:
    def __init__(self, models_dir):
        import tensorflow as tf  # only needed for interpreter
        meta_path = os.path.join(models_dir, "meta.json")
        with open(meta_path) as f: meta = json.load(f)
        self.W = int(meta["window"])
        self.F = int(meta["feat_dim"])
        self.mu = np.asarray(meta["norm_mean"], dtype=np.float32)
        self.sd = np.asarray(meta["norm_std"], dtype=np.float32)
        self.interp = tf.lite.Interpreter(model_path=os.path.join(models_dir, "gesture_small_fp32.tflite"))
        self.interp.allocate_tensors()
        sigs = self.interp.get_signature_list()
        # Use default signature
        self.run = self.interp.get_signature_runner(list(sigs.keys())[0])

        # Cache input/output names from the signature
        input_details = self.run.get_input_details()
        output_details = self.run.get_output_details()
        self.input_name = list(input_details.keys())[0]

        # Prefer semantic names if present; fall back to shape-based mapping
        names = set(output_details.keys())
        out_map = {}
        if {'tracks','volume','playpause'}.issubset(names):
            out_map['tracks'] = 'tracks'
            out_map['volume'] = 'volume'
            out_map['playpause'] = 'playpause'
        else:
            for name, detail in output_details.items():
                try:
                    last = int(detail['shape'][-1])
                except Exception:
                    last = None
                if last == 3:
                    if 'tracks' not in out_map: out_map['tracks'] = name
                    elif 'volume' not in out_map: out_map['volume'] = name
                elif last == 1:
                    out_map['playpause'] = name
        if len(out_map) < 3:
            raise RuntimeError(f"Couldn't map all TFLite output heads. Found: {list(output_details.keys())}")
        self.out_map = out_map

    def predict(self, win_arr):
        # win_arr: [W, F_raw]. If your per-frame vector > F, slice first F.
        X = win_arr[:, :self.F].astype(np.float32)
        X = (X - self.mu) / self.sd
        Xb = X[None, ...]  # [1,W,F]
        outs = self.run(**{self.input_name: Xb})
        
        # In train_nn.py, outputs are named 'tracks', 'volume', 'playpause'
        p_tr = outs[self.out_map['tracks']][0]
        p_vl = outs[self.out_map['volume']][0]
        p_pp = float(outs[self.out_map['playpause']][0][0])
        return p_tr, p_vl, p_pp

# ---------------- Main loop ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--backend", choices=["xgb","tflite","shadow"], default="tflite")
    args = ap.parse_args()

    # Load meta for window size (works for both backends; fallback for XGB)
    W_meta = None
    meta_path = os.path.join(args.models, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f: W_meta = json.load(f).get("window")

    # Backends
    xgb = None; tfl = None
    if args.backend in ("xgb","shadow"):
        # If no meta, read W from feat_schema.json (XGB trainer)
        if W_meta is None:
            with open(os.path.join(args.models,"feat_schema.json")) as f:
                W_meta = json.load(f)["window"]
        xgb = XGBBackend(args.models, window=W_meta)
    if args.backend in ("tflite","shadow"):
        tfl = TFLiteBackend(args.models)
        W_meta = tfl.W

    # Camera + buffers
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

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
            frame = cv2.flip(frame, 1)
            now = time.time()
            info = ""

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label
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

                    # Backend predictions
                    p_tr_t, p_vl_t, p_pp_t = (None, None, None)
                    if tfl:   p_tr_t, p_vl_t, p_pp_t = tfl.predict(win_arr)
                    if xgb:
                        p_tr_x, p_vl_x, p_pp_x = xgb.predict(win_arr)

                    # Choose per backend
                    if args.backend == "xgb":
                        p_tr, p_vl, p_pp = p_tr_x, p_vl_x, (p_pp_x if xgb.play is not None else None)
                    elif args.backend == "tflite":
                        p_tr, p_vl, p_pp = p_tr_t, p_vl_t, p_pp_t
                    else:  # shadow: prefer TFLite for actions, log disagreements
                        p_tr, p_vl, p_pp = p_tr_t, p_vl_t, p_pp_t
                        try:
                            # simple disagreement log
                            if (np.argmax(p_tr_t) != np.argmax(p_tr_x)) or (np.argmax(p_vl_t) != np.argmax(p_vl_x)):
                                print(f"[shadow] tracks tfl={np.argmax(p_tr_t)} xgb={np.argmax(p_tr_x)} | "
                                      f"vol tfl={np.argmax(p_vl_t)} xgb={np.argmax(p_vl_x)}")
                        except Exception:
                            pass

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
                cv2.putText(frame, mode, (10,24), 0, 0.7, (255,255,255), 2)
            else:
                prev_state=None; win = WinBuf(W_meta or 10)

            # Overlay debug: show play/pause probability if available
            try:
                if tfl:
                    cv2.putText(frame, f"PP: {p_pp_t:.2f}", (10, 76), 0, 0.6, (0,200,255), 2)
            except Exception:
                pass

            if info:
                cv2.putText(frame, info, (10, 50), 0, 0.8, (0,255,0), 2)
            cv2.putText(frame, "q = quit", (10, h-12), 0, 0.6, (200,200,200), 2)

            cv2.imshow('Gesture — Inference (XGB/TFLite)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
