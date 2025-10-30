"""
Train a tiny GRU multi-head model on landmark feature windows and export to TFLite.

Usage:
  python -m pip install --upgrade pip
  pip install tensorflow==2.15.0 tqdm pandas numpy scikit-learn
  python train_nn.py --glob "data/*.csv" --outdir models --window 10

Outputs:
  models/gesture_small_fp32.tflite
  models/meta.json   (contains W, F, normalization mean/std, label maps)
"""

import os, sys, glob, json, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRACK_LABELS = {'NONE':0,'NEXT':1,'PREV':2}
VOL_LABELS   = {'VOL_DOWN':0,'NONE':1,'VOL_UP':2}
# PLAY/PAUSE is binary: 0=OTHER, 1=PLAY_PAUSE

def load_sessions(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\n❌ No CSVs for pattern: {pattern}\nRecord some data first.", file=sys.stderr)
        sys.exit(1)
    print("Found:"); [print(" •", fp) for fp in files]
    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp))
        except Exception as e:
            print(f"Warning: cannot read {fp}: {e}", file=sys.stderr)
    if not frames:
        print("❌ Could not read any CSVs.", file=sys.stderr); sys.exit(1)
    df = pd.concat(frames, ignore_index=True)

    needed = {'session','has_hand','label_name'}
    miss = needed - set(df.columns)
    if miss:
        print(f"❌ Missing columns: {sorted(miss)}", file=sys.stderr); sys.exit(1)

    feat_cols_all = [c for c in df.columns if c.startswith('f')]
    if not feat_cols_all:
        print("❌ No feature columns starting with 'f'.", file=sys.stderr); sys.exit(1)

    # Detect usable prefix (like your XGB trainer)
    nonnull_ratio = df[feat_cols_all].notna().mean(axis=0).values
    k = 0
    for i, r in enumerate(nonnull_ratio):
        if r >= 0.8: k = i+1
        else: break
    if k < 5:
        for i, r in enumerate(nonnull_ratio):
            if r >= 0.5: k = i+1
            else: break
    if k < 5:
        print("❌ Could not determine stable feature prefix.", file=sys.stderr); sys.exit(1)

    feat_cols = feat_cols_all[:k]
    print(f"Using first {k} feature columns: {feat_cols[0]}..{feat_cols[-1]}")

    before = len(df)
    any_feat = df[feat_cols].notna().any(axis=1)
    df = df[(df['has_hand']==1) & any_feat].copy()
    df = df.dropna(subset=feat_cols)
    after = len(df)
    print(f"Frames with hands: {after:,} (dropped {before-after:,})")
    if after == 0:
        print("❌ Zero usable frames after filtering.", file=sys.stderr); sys.exit(1)

    return df, feat_cols

def make_windows(df, feat_cols, W=10, step=1):
    """Build sequence windows [N, W, F] and labels for three heads."""
    X, y_tr, y_vl, y_pp = [], [], [], []
    for sess, g in tqdm(df.groupby('session'), desc="Building windows", unit="session"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < W: continue
        F = g[feat_cols].values.astype(np.float32)
        L = g['label_name'].values
        A = g['twirl_ddeg'].values.astype(np.float32) if 'twirl_ddeg' in g.columns else np.zeros(n, np.float32)

        for i in range(0, n - W + 1, step):
            winF = F[i:i+W]                 # [W, F]
            winL = L[i:i+W]
            winA = A[i:i+W]
            X.append(winF)

            # Tracks: majority NEXT/PREV/NONE
            counts = {k: int(np.sum(winL==k)) for k in TRACK_LABELS.keys()}
            lab_tr = max(counts, key=counts.get)
            y_tr.append(TRACK_LABELS[lab_tr])

            # Volume: sign of cumulative angle (or your hotkey labels if preferred)
            ang_sum = float(winA.sum())
            if ang_sum > 5.0:  y_vl.append(VOL_LABELS['VOL_UP'])
            elif ang_sum < -5.0: y_vl.append(VOL_LABELS['VOL_DOWN'])
            else: y_vl.append(VOL_LABELS['NONE'])

            # Play/Pause: majority PLAY_PAUSE
            pp = int(np.sum(winL=='PLAY_PAUSE')) > (W//2)
            y_pp.append(1 if pp else 0)

    if not X:
        print("❌ No windows created. Record longer sessions.", file=sys.stderr); sys.exit(1)

    X = np.stack(X)                 # [N, W, F]
    y_tr = np.array(y_tr)           # [N]
    y_vl = np.array(y_vl)           # [N]
    y_pp = np.array(y_pp)           # [N]
    return X, y_tr, y_vl, y_pp

def build_model(W, F):
    inp = keras.Input(shape=(W, F), name="seq")
    x = layers.LSTM(64, return_sequences=True, unroll=True)(inp)
    x = layers.LSTM(64, unroll=True)(x)
    x = layers.Dense(64, activation="relu")(x)
    t = layers.Dense(3, activation="softmax", name="tracks")(x)
    v = layers.Dense(3, activation="softmax", name="volume")(x)
    p = layers.Dense(1, activation="sigmoid", name="playpause")(x)
    model = keras.Model(inp, [t, v, p])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"tracks":"sparse_categorical_crossentropy",
              "volume":"sparse_categorical_crossentropy",
              "playpause":"binary_crossentropy"},
        loss_weights={"tracks":1.0, "volume":1.0, "playpause":0.7},
        metrics={"tracks":"accuracy","volume":"accuracy","playpause":"accuracy"},
    )
    return model

def export_tflite(model, out_path, rep_data=None):
    import tensorflow as tf, tempfile, sys

    def _configure(conv, use_int8):
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        conv._experimental_lower_tensor_list_ops = False
        try:
            conv.experimental_enable_resource_variables = True
        except Exception:
            pass
        try:
            conv.experimental_new_converter = True
        except Exception:
            pass
        if use_int8:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = rep_data
        return conv

    # Always convert via SavedModel to avoid Keras private API differences
    with tempfile.TemporaryDirectory() as tmp:
        # Keras 3 exports SavedModel via model.export(); fall back for older Keras
        try:
            export_fn = getattr(model, "export", None)
            if callable(export_fn):
                export_fn(tmp)
            else:
                model.save(tmp, include_optimizer=False)
        except Exception:
            model.save(tmp, include_optimizer=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conv = _configure(conv, rep_data is not None)
        tfl = conv.convert()

    open(out_path, "wb").write(tfl)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. 'data/*.csv'")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, feat_cols = load_sessions(args.glob)
    X, y_tr, y_vl, y_pp = make_windows(df, feat_cols, W=args.window, step=1)
    N, W, F = X.shape
    print(f"Dataset: N={N}, W={W}, F={F}")

    # train/val split (random is fine if sessions are mixed; session-wise split is stricter)
    Xtr, Xva, tr_tr, tr_va = train_test_split(X, y_tr, test_size=args.val_split, random_state=42, stratify=y_tr)
    Xtr2, Xva2, vl_tr, vl_va = train_test_split(X, y_vl, test_size=args.val_split, random_state=42, stratify=y_vl)
    Xtr3, Xva3, pp_tr, pp_va = train_test_split(X, y_pp, test_size=args.val_split, random_state=42, stratify=y_pp)
    # keep shapes consistent
    assert Xtr.shape == Xtr2.shape == Xtr3.shape

    # Per-feature normalization (mean/std over training set)
    mu = Xtr.mean(axis=(0,1))      # [F]
    sd = Xtr.std(axis=(0,1)) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    model = build_model(W, F)
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]
    hist = model.fit(
        Xtr, {"tracks":tr_tr, "volume":vl_tr, "playpause":pp_tr},
        validation_data=(Xva, {"tracks":tr_va, "volume":vl_va, "playpause":pp_va}),
        epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=1
    )

    # Export FP32 TFLite
    tfl_path = os.path.join(args.outdir, "gesture_small_fp32.tflite")
    export_tflite(model, tfl_path)
    print(f"Saved TFLite: {tfl_path}")

    # Save meta (normalization + dims + label maps)
    meta = {
        "window": W,
        "feat_dim": F,
        "norm_mean": mu.tolist(),
        "norm_std": sd.tolist(),
        "label_maps": {
            "tracks": TRACK_LABELS,
            "volume": VOL_LABELS,
            "playpause": {"OTHER":0, "PLAY_PAUSE":1}
        }
    }
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta: {meta_path}")
