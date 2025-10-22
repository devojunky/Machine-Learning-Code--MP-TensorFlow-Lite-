# Step 2 of 3 Train gesture recognition models
# Example usage: python train_models.py --glob "data/*.csv" --outdir models
# you can use the --overwrite flag but lowkey that was just for debugging u should probalbly not use it if u care about ur models
# This works pretty well for like 3 mins of training in total. More data is probably better but its whatever. Training "none" is very important
# Daniel Liang dipping out ✌️

import glob, json, argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm

TRACK_LABELS = {'NONE': 0, 'NEXT': 1, 'PREV': 2}
VOL_LABELS   = {'VOL_DOWN': 0, 'NONE': 1, 'VOL_UP': 2}
PLAY_LABELS  = {'OTHER': 0, 'PLAY_PAUSE': 1}

def save_safe(obj, path, overwrite=False):
    if overwrite or not os.path.exists(path):
        joblib.dump(obj, path); return path
    base, ext = os.path.splitext(path)
    ts = time.strftime('%Y%m%d-%H%M%S')
    newp = f"{base}.{ts}{ext}"
    joblib.dump(obj, newp)
    return newp

def load_sessions(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\n❌ No CSV files found for pattern: {pattern}\n"
              "   Tip: Run the recorder first, e.g.\n"
              "     python record_and_label.py --out data/session1.csv\n")
        sys.exit(1)
    print("Found:"); [print(" •", fp) for fp in files]
    frames_list = []
    for fp in files:
        try: frames_list.append(pd.read_csv(fp))
        except Exception as e: print(f"Warning: failed to read {fp}: {e}", file=sys.stderr)
    if not frames_list:
        print("❌ Could not read any CSVs.", file=sys.stderr); sys.exit(1)
    df = pd.concat(frames_list, ignore_index=True)

    needed = {'session','has_hand','label_name'}
    miss = needed - set(df.columns)
    if miss:
        print(f"❌ Missing required columns: {sorted(miss)}", file=sys.stderr); sys.exit(1)

    feat_cols_all = [c for c in df.columns if c.startswith('f')]
    if not feat_cols_all:
        print("❌ No feature columns starting with 'f'.", file=sys.stderr); sys.exit(1)

    # Detect usable prefix
    nonnull_ratio = df[feat_cols_all].notna().mean(axis=0).values
    k = 0
    for idx, r in enumerate(nonnull_ratio):
        if r >= 0.8: k = idx+1
        else: break
    if k < 5:
        for idx, r in enumerate(nonnull_ratio):
            if r >= 0.5: k = idx+1
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
    print(f"Loaded {after:,} frames with hands (dropped {before-after:,})")
    if after == 0:
        print("❌ Zero usable frames after filtering.", file=sys.stderr); sys.exit(1)
    return df, feat_cols

def make_windows(df, feat_cols, W=10, step=1):
    X_trk, y_trk, X_vol, y_vol, X_play, y_play = [], [], [], [], [], []
    for sess, g in tqdm(df.groupby('session'), desc="Building windows", unit="session"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < W: continue
        feats = g[feat_cols].values.astype(np.float32)
        labels = g['label_name'].values
        twstep = g['twirl_ddeg'].values.astype(np.float32) if 'twirl_ddeg' in g.columns else np.zeros(n, np.float32)

        for i in range(0, n - W + 1, step):
            winF = feats[i:i+W]
            winL = labels[i:i+W]
            winA = twstep[i:i+W]

            mean = winF.mean(axis=0)
            std  = winF.std(axis=0)
            dxy  = winF[-1, :2] - winF[0, :2]
            ang_sum = float(winA.sum())
            xvec = np.concatenate([mean, std, dxy, [ang_sum]]).astype(np.float32)

            # TRACK majority
            lab_counts = {k: int(np.sum(winL==k)) for k in TRACK_LABELS.keys()}
            lab = max(lab_counts, key=lab_counts.get)
            y_trk.append(TRACK_LABELS[lab]); X_trk.append(xvec)

            # VOLUME by sign of angle
            if ang_sum > 5.0: y_vol.append(2)      # VOL_UP
            elif ang_sum < -5.0: y_vol.append(0)   # VOL_DOWN
            else: y_vol.append(1)                  # NONE
            X_vol.append(xvec)

            # PLAY/PAUSE majority
            pp = int(np.sum(winL=='PLAY_PAUSE')) > (W//2)
            y_play.append(1 if pp else 0)
            X_play.append(xvec)

    if not X_trk or not X_vol or not X_play:
        print("❌ No usable windows created. Check your recordings.", file=sys.stderr); sys.exit(1)

    return (np.stack(X_trk), np.array(y_trk),
            np.stack(X_vol), np.array(y_vol),
            np.stack(X_play), np.array(y_play))

def build_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            max_depth=4, n_estimators=250, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective='multi:softprob', n_jobs=-1))
    ])

def build_bin_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            max_depth=3, n_estimators=220, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective='binary:logistic', n_jobs=-1))
    ])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--outdir', default='models')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--overwrite', action='store_true', help='overwrite existing model files')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, feat_cols = load_sessions(args.glob)
    Xtr, ytr, Xv, yv, Xp, yp = make_windows(df, feat_cols, W=args.window, step=1)
    print(f"\nWindows built: tracks={len(Xtr):,}, volume={len(Xv):,}, play={len(Xp):,}")

    # splits
    Xtr_tr, Xtr_va, ytr_tr, ytr_va = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
    Xv_tr,  Xv_va,  yv_tr,  yv_va  = train_test_split(Xv,  yv,  test_size=0.2, random_state=42, stratify=yv)
    Xp_tr,  Xp_va,  yp_tr,  yp_va  = train_test_split(Xp,  yp,  test_size=0.2, random_state=42, stratify=yp)

    print("\nTraining models...")
    tracks = build_clf(); tracks.fit(Xtr_tr, ytr_tr)
    volcls = build_clf(); volcls.fit(Xv_tr, yv_tr)
    play   = build_bin_clf(); play.fit(Xp_tr, yp_tr)

    print(f"Tracks val acc: {tracks.score(Xtr_va, ytr_va):.3f}")
    print(f"Volume val acc: {volcls.score(Xv_va, yv_va):.3f}")
    print(f"Play/Pause val acc: {play.score(Xp_va, yp_va):.3f}")

    p1 = save_safe(tracks, os.path.join(args.outdir, 'tracks_xgb.joblib'), args.overwrite)
    p2 = save_safe(volcls, os.path.join(args.outdir, 'vol_cls_xgb.joblib'), args.overwrite)
    p3 = save_safe(play,   os.path.join(args.outdir, 'play_xgb.joblib'),    args.overwrite)

    with open(os.path.join(args.outdir, 'feat_schema.json'), 'w') as f:
        json.dump({"feat_count": len(feat_cols), "window": args.window}, f, indent=2)

    print(f"\n✅ Saved:\n  {p1}\n  {p2}\n  {p3}\n  {os.path.join(args.outdir,'feat_schema.json')}")

# another amazing daniel liang production 🚀🚀🚀