import cv2, time, math, csv, argparse, os
import numpy as np
import mediapipe as mp

# --- pynput has been removed as it conflicts with the OpenCV event loop on macOS ---

SEL = [0,1,2,3,4,5,9,13,17,6,8,10,12,14,16,18,20]

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def _rot2d(pt, cs, sn):
    x,y = pt
    return np.array([cs*x - sn*y, sn*x + cs*y], dtype=np.float32)

def _signed_angle(a, b):
    A = math.atan2(a[1], a[0]); B = math.atan2(b[1], b[0])
    d = B - A
    if d > math.pi:  d -= 2*math.pi
    if d < -math.pi: d += 2*math.pi
    return d

class KnobTrail:
    def __init__(self, n=24):
        self.n=n; self.buf=[]; self.center=None; self.last_vec=None
    def reset(self):
        self.buf.clear(); self.center=None; self.last_vec=None
    def update(self, x,y):
        self.buf.append((x,y))
        if len(self.buf) > self.n: self.buf.pop(0)
        if len(self.buf) < 6: self.center=None; self.last_vec=None; return 0.0, False
        pts = np.array(self.buf, dtype=np.float32)
        c = pts.mean(axis=0); self.center=c
        v = pts[-1] - c
        r = float(np.linalg.norm(v)) + 1e-9
        if r < 0.03: self.last_vec=None; return 0.0, False
        v = v / r
        ddeg=0.0
        if self.last_vec is not None:
            d = _signed_angle(self.last_vec, v)
            d = -d
            if abs(d) < math.radians(0.8): d = 0.0
            ddeg = math.degrees(d)
        self.last_vec = v
        return ddeg, True

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
    def curl(tip, pip): return C[tip,1] - C[pip,1]
    curls = np.array([curl(4,3), curl(8,6), curl(12,10), curl(16,14), curl(20,18)], dtype=np.float32)
    thumb_dir = C[4] - C[3]; thumb_dir = thumb_dir / (np.linalg.norm(thumb_dir)+1e-9)
    nz = np.cross(P[5], P[17])[2:3]
    if prev_state is None:
        d_idx = np.zeros(2, np.float32); d_td = np.zeros(2, np.float32)
    else:
        d_idx = C[8] - prev_state['idx']; d_td = thumb_dir - prev_state['td']
    hflag = np.array([1.0 if handed_label == 'Right' else 0.0], dtype=np.float32)
    feat = np.concatenate([coords, curls, thumb_dir, nz, d_idx, d_td, hflag]).astype(np.float32)
    state = {'idx': C[8].copy(), 'td': thumb_dir.copy()}
    return feat, state, C

# Labels map (char -> (name, id))
LABELS_CHAR = {
    '0': ('NONE', 0),
    '1': ('NEXT', 1),
    '2': ('PREV', 2),
    '3': ('VOL_UP', 3),
    '4': ('VOL_DOWN', 4),
    '5': ('PLAY_PAUSE', 5),
}
INSTR = "Press: 1 NEXT | 2 PREV | 3 VOL_UP | 4 VOL_DOWN | 5 PLAY/PAUSE | 0 NONE | q save+quit"

# --- State for current label is now managed directly in the main loop ---
# --- All pynput related code (_pressed, _pressed_lock, _on_press, _on_release, _current_label) has been removed ---

def record(out_csv):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    trail = KnobTrail(24)
    prev_state=None
    
    # --- New: state variable for the current label ---
    current_label_name = 'NONE'
    current_label_id = 0

    # open for append; header only if new/empty
    new_file = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    f = open(out_csv, 'a', newline='')
    writer = csv.writer(f)
    if new_file:
        writer.writerow(['session','t','label_name','label_id','has_hand','twirl_ddeg'] + [f'f{i}' for i in range(200)])

    session = os.path.splitext(os.path.basename(out_csv))[0]
    t0 = time.time()

    # --- keyboard listener has been removed ---

    # compatibility alias
    mp_hands.HANDS = mp_hands.Hands

    with mp_hands.HANDS(static_image_mode=False,max_num_hands=1,model_complexity=0,
                        min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        while True:
            # --- quit flag is no longer needed, we check for 'q' from cv2.waitKey ---
            ok, frame = cap.read()
            if not ok: break
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            t = time.time() - t0
            has_hand = 0
            twirl_ddeg=0.0

            if res.multi_hand_landmarks:
                has_hand = 1
                hand = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(), mp_style.get_default_hand_connections_style())

                feat, prev_state, C = extract_features(hand.landmark, handed, prev_state)
                ddeg, active = trail.update(C[8,0], C[8,1])
                twirl_ddeg = ddeg if active else 0.0
                row_feat = feat.tolist()
            else:
                prev_state=None; trail.reset(); row_feat = []

            # The key press handling is now done via cv2.waitKey below
            writer.writerow([session, f"{t:.3f}", current_label_name, current_label_id, has_hand, f"{twirl_ddeg:.3f}"] + row_feat)

            # HUD
            cv2.putText(frame, f"Recording: {session}", (10,24), 0, 0.7, (255,255,255), 2)
            cv2.putText(frame, INSTR, (10,50), 0, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Current label (press key): {current_label_name}", (10,76), 0, 0.7, (0,255,0), 2)
            cv2.putText(frame, "Press q to save+quit", (10,102), 0, 0.6, (200,200,200), 2)
            cv2.imshow('Record & Label (press-to-label)', frame)

            # Use cv2.waitKey to handle all key presses
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            
            key_char = chr(k) if k != 255 else None # 255 is returned when no key is pressed
            if key_char and key_char in LABELS_CHAR:
                new_name, new_id = LABELS_CHAR[key_char]
                if new_id == current_label_id:
                    # Toggle off if pressing the same key again
                    current_label_name = 'NONE'
                    current_label_id = 0
                else:
                    # Set the new label
                    current_label_name = new_name
                    current_label_id = new_id


    # cleanup
    f.close()
    cap.release(); cv2.destroyAllWindows()
    print(f"Appended to: {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='output CSV path, e.g., data/session1.csv')
    args = parser.parse_args()
    record(args.out)
