## Gesture Control with MediaPipe + TensorFlow Lite (macOS)

Real-time hand-gesture controls for media playback using MediaPipe hand landmarks. Supports two inference backends:

- **TFLite**: small LSTM multi-head model exported as `.tflite`
- **XGBoost**: classic features + tree models

Works on macOS and uses AppleScript to control Spotify or Music. Recording, training, and inference are all included.

---

### Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv tfenv
source tfenv/bin/activate
python -m pip install --upgrade pip

# 2) Install dependencies
pip install -r requirements.txt

# 3) Record labeled data (press keys while recording)
python record_and_label.py --out data/session1.csv

# 4a) Train classic models (XGBoost)
python train_models.py --glob "data/*.csv" --outdir models

# 4b) Train neural net and export TFLite
python train_nn.py --glob "data/*.csv" --outdir models --window 10

# 5) Run real-time inference (default: TFLite)
python run_inference.py --models models --backend tflite
# other options: --backend xgb | shadow
```

---

### Requirements

- macOS with camera access (tested on Apple Silicon).
- Python 3.11 recommended.
- OpenCV, MediaPipe, TensorFlow (macOS build), XGBoost. All pinned in `requirements.txt`.

If you run into platform wheels issues on Apple Silicon, prefer the provided `requirements.txt`. Ensure command-line tools are installed: `xcode-select --install`.

---

### Recording Data

Run the recorder and label in real time:

```bash
python record_and_label.py --out data/session1.csv
```

- Window title: “Record & Label (press-to-label)”
- Keys while recording:
  - `1`: NEXT
  - `2`: PREV
  - `3`: VOL_UP
  - `4`: VOL_DOWN
  - `5`: PLAY_PAUSE
  - `0`: NONE (neutral)
  - `q`: save and quit
- Pressing the same key toggles the label off (back to NONE).
- Output CSV columns: `session, t, label_name, label_id, has_hand, twirl_ddeg, f0..f199` (unused feature slots are left blank).

Tips:

- Record multiple short sessions (mix “NONE” generously to reduce false positives).
- Keep the hand in frame; palm width normalization and orientation are computed per frame.

---

### Training (XGBoost)

```bash
python train_models.py --glob "data/*.csv" --outdir models
```

Outputs in `models/`:

- `tracks_xgb.joblib` (NEXT/PREV/NONE)
- `vol_cls_xgb.joblib` (VOL_UP/VOL_DOWN/NONE)
- `play_xgb.joblib` (PLAY_PAUSE binary)
- `feat_schema.json` (feature count and window size)

---

### Training (Neural Network → TFLite)

Train a tiny LSTM multi-head classifier on windows of raw per-frame features and export to TFLite.

```bash
python train_nn.py --glob "data/*.csv" --outdir models --window 10
```

Outputs in `models/`:

- `gesture_small_fp32.tflite`
- `meta.json` (window, feature dim, normalization mean/std, label maps)

You may also use the sample model in `tf_models/` by pointing `--models tf_models` when running inference.

---

### Run Real-Time Inference

```bash
# TFLite backend (recommended)
python run_inference.py --models models --backend tflite

# XGBoost backend
python run_inference.py --models models --backend xgb

# Shadow mode: run both, act on TFLite, log disagreements
python run_inference.py --models models --backend shadow
```

Notes:

- Press `q` to quit.
- On macOS, AppleScript is used to control Spotify or Music. Ensure one of them is running for media controls to work.
- If using `--backend xgb` and no `meta.json` is present, the script falls back to `feat_schema.json` from the XGB training step.

---

### Gestures (default mapping)

- **Track control (index finger down)**
  - Next track: point thumb right, palm closed
  - Previous track: point thumb left, palm closed
- **Volume control (index finger up)**
  - Volume up: twirl index finger clockwise
  - Volume down: twirl index finger counter-clockwise
- **Play / Pause (either mode)**
  - Shaka 🤙

Runtime behavior (simplified):

- Play/Pause triggers when its probability exceeds a threshold for a few consecutive frames.
- Volume vs Tracks mode gated by index-up posture and palm orientation.

You can tweak thresholds like `PP_THRESH`, hold counts, and cooldowns inside `run_inference.py`.

---

### Repository Layout

- `record_and_label.py`: Capture webcam, label frames with hotkeys, write CSV.
- `train_models.py`: Build classic features over windows, train XGBoost models.
- `train_nn.py`: Build windowed sequences, train LSTM multi-head model, export `.tflite` and `meta.json`.
- `run_inference.py`: Real-time MediaPipe + backend(s), sends OS media commands asynchronously.
- `data/`: Your recorded CSVs.
- `models/`: Trained artifacts and metadata.
- `tf_models/`: Example TFLite model and meta for quick testing.

---

### Troubleshooting

- **Camera not opening**: Grant Terminal/IDE camera permission in System Settings → Privacy & Security → Camera.
- **High CPU/GPU usage**: Lower webcam resolution in `run_inference.py` or close other GPU-heavy apps.
- **No media action**: Ensure Spotify or Music is running. AppleScript control is macOS-only.
- **XGB only working**: Verify `models/feat_schema.json` exists (created by `train_models.py`). For TFLite, ensure `models/meta.json` and `gesture_small_fp32.tflite` exist.
- **TensorFlow install issues**: Use the provided `requirements.txt`; ensure `xcode-select --install` is completed; try a clean venv.

---

### License & Credits

Created by Daniel Liang. See source headers for notes and credits.
