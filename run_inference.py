# Step 3 of 3 Run gesture recognition inference
# Example usage: python run_inference.py --models models
# This works for macOS only since it uses apple script to control Music/Spotify only
# It's multithreaded so it doesn't mess with inference bc idk why but subprocess calls can be slow and freeze the UI
# prolly hard coded limitation from apple script or something
# I tested this on macOS 26 and Apple M1 Max. My system was taking up 35 watts during inference with webcam
# But that was running at 960x540 and also running spotify and streaming at the same time we could optimize it fs
# I'm praying this runs on a Raspberry Pi but idk if inference will do well on rasp
# I hope it runs tho 🙏🙏🙏🙏
# I also tried hardcoding the gestures and it really didn't work well. Lots of false positives.
# ML is wayyyy better
# In the future we could add self correction or feedback so you can teach it your own gestures, but that's out of scope for now
# Daniel Liang signing out ✌️

# ------ Gestures ------
# Track control (index finger down):
#   Next track:      poing thumb right, palm closed
#   Previous track:  point thumb left, palm closed
# Volume control (index finger up):
#   Volume up:       twirl index fingers clockwise
#   Volume down:     twirl index fingers counter-clockwise
# Play / Pause (either mode):
#   Shakaaaaaa 🤙🤙🤙

import cv2, time, math, json, argparse, os, threading, queue, sys
import numpy as np
import mediapipe as mp
import joblib
import subprocess

# Palm gating for index_up (prevents false volume mode when back of hand faces camera)
NZ_THRESH = 0.02          # raise to 0.04–0.08 so its stricter, but this works good i think??
PALM_FRONT_SIGN = -1      # -1 means palm-front if nz < 0 (flip to +1 if it feels inverted)
# Feature layout constants (match extract_features order)
COORDS_LEN = 34           # 17 landmarks * 2
CURLS_LEN  = 5
TD_LEN     = 2
NZ_POS     = COORDS_LEN + CURLS_LEN + TD_LEN  # index of nz inside feat vector


# ---------------- Media command worker (non-blocking) ----------------
class CommandWorker(threading.Thread):
    def __init__(self, max_queue=64):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=max_queue)

    def run(self):
        while True:
            fn, args = self.q.get()
            try:
                fn(*args)
            except Exception as e:
                print("[cmd worker]", e, file=sys.stderr)
            finally:
                self.q.task_done()

    def submit(self, fn, *args):
        try:
            self.q.put_nowait((fn, args))
        except queue.Full:
            # Drop if overwhelmed; avoids building backlog
            pass

worker = CommandWorker(); worker.start()

def osa(script):
    # Blocking in worker thread, not UI thread
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True)

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

def async_toggle_play_pause(): worker.submit(_osa_toggle_playpause)
def async_next_track(): worker.submit(_osa_next)
def async_prev_track(): worker.submit(_osa_prev)
def async_nudge_vol(delta): worker.submit(_osa_nudge, delta)

# ---------------- MediaPipe & features (must match training and include all models in the folder) ----------------
SEL = [0,1,2,3,4,5,9,13,17,6,8,10,12,14,16,18,20]
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def _rot2d(pt, cs, sn):
    x,y = pt
    return np.array([cs*x - sn*y, sn*x + cs*y], dtype=np.float32)

class WinBuf:
    def __init__(self, W): self.W=W; self.buf=[]
    def add(self, x): self.buf.append(x);  self.buf=self.buf[-self.W:]
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

def make_window_features(win_arr):
    """
    Match training: [mean(F), std(F), dxy(index_tip), ang_sum_degrees]
    """
    import math
    X = win_arr
    mean = X.mean(axis=0)
    std  = X.std(axis=0)

    # index tip xy are inside coords block; SEL.index(8)=10 -> coords offsets [20,21]
    INDEX_IN_SEL = SEL.index(8)
    idx_xy_offset = 2 * INDEX_IN_SEL
    idx_xy_end    = idx_xy_offset + 2
    idx_series = X[:, idx_xy_offset:idx_xy_end]  # [W,2]

    dxy = idx_series[-1] - idx_series[0]

    # cumulative signed angle (deg) around window centroid (y-down -> flip sign)
    c = idx_series.mean(axis=0)
    vecs = idx_series - c
    norms = (np.linalg.norm(vecs, axis=1) + 1e-9)[:, None]
    v = vecs / norms
    ang_sum = 0.0
    for t in range(1, v.shape[0]):
        ax, ay = v[t-1]
        bx, by = v[t]
        da = math.atan2(by, bx) - math.atan2(ay, ax)
        if da > math.pi:  da -= 2*math.pi
        if da < -math.pi: da += 2*math.pi
        ang_sum += -da
    ang_sum_deg = math.degrees(ang_sum)

    return np.concatenate([mean, std, dxy, [ang_sum_deg]]).astype(np.float32)

# ---------------- Main ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='models')
    args = ap.parse_args()

    tracks = joblib.load(os.path.join(args.models, 'tracks_xgb.joblib'))
    volcls = joblib.load(os.path.join(args.models, 'vol_cls_xgb.joblib'))
    play   = joblib.load(os.path.join(args.models, 'play_xgb.joblib'))
    with open(os.path.join(args.models, 'feat_schema.json')) as f:
        schema = json.load(f)
    W = schema.get('window', 10)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    prev_state=None
    win = WinBuf(W)
    cooldown_track = 0.0
    cooldown_vol   = 0.0
    cooldown_pause = 0.0
    hold_next=hold_prev=0
    LOCK_AFTER_PAUSE = 0.4    # small lock so other actions dont piggyback. that would be very smart but also very dangerous

    cap.release(); cv2.destroyAllWindows()
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='models')
    args = ap.parse_args()

    tracks = joblib.load(os.path.join(args.models, 'tracks_xgb.joblib'))
    volcls = joblib.load(os.path.join(args.models, 'vol_cls_xgb.joblib'))
    with open(os.path.join(args.models, 'feat_schema.json')) as f:
        schema = json.load(f)
    W = schema.get('window', 10)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    prev_state=None
    win = WinBuf(W)
    cooldown_track = 0.0
    cooldown_vol   = 0.0
    hold_next=hold_prev=0

    with mp_hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=0,
                        min_detection_confidence=0.5,min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb) # this is crazy heavy computatoin, can be optimized, might be slow on rpi as is
            now = time.time()
            info = ""

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, # PLEASE REMOVE THIS IN PROD PLEASE PLEASE DONT FORGET BC PI WILL RUN HEADLESS thanks 🙏👍
                    mp_style.get_default_hand_landmarks_style(), mp_style.get_default_hand_connections_style())
                feat, prev_state, C = extract_features(hand.landmark, handed, prev_state)
                win.add(feat)

                # gate by palm orientation:
                nz = float(feat[NZ_POS])  # palm normal z from extract_features
                palm_front = (PALM_FRONT_SIGN * nz) > NZ_THRESH
                index_up = (C[8,1] < C[6,1]) and palm_front # palm_front might be redundant. idrk, need more experiments. fml


                if win.ready():
                    x = make_window_features(win.stack())
                    # --- shaka 🤙 / play-pause first ---
                    p_play = play.predict_proba([x])[0][1]
                    if p_play > 0.70 and now > cooldown_pause:
                        async_toggle_play_pause()
                        info = '⏯️ Play/Pause'
                        cooldown_pause = now + 0.9
                        cooldown_track = now + LOCK_AFTER_PAUSE
                        cooldown_vol   = now + LOCK_AFTER_PAUSE
                    else:
                        # normal pipeline
                        p_tr = tracks.predict_proba([x])[0]  # [NONE,NEXT,PREV]
                        p_vl = volcls.predict_proba([x])[0]  # [DOWN,NONE,UP]

                        if index_up and now > cooldown_vol:
                            if p_vl[2] > 0.65:
                                async_nudge_vol(+1); info = '🔊 Vol +'; cooldown_vol = now + 0.12
                            elif p_vl[0] > 0.65:
                                async_nudge_vol(-1); info = '🔉 Vol −'; cooldown_vol = now + 0.12
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
                prev_state=None; win = WinBuf(W)

            if info:
                cv2.putText(frame, info, (10, 50), 0, 0.8, (0,255,0), 2)
            cv2.putText(frame, "q = quit", (10, h-12), 0, 0.6, (200,200,200), 2)

            cv2.imshow('Gesture ML — Inference (non-blocking)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

# another amazing daniel liang production 🚀🚀🚀