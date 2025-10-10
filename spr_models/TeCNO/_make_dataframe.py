# make_dataframe.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------- Path settings ---------
ROOT        = Path("/data2/local_datasets/cholec80")
KAYOUNG_ROOT= Path("/data2/local_datasets/cholec80_kayoung")
IMG_ROOT    = ROOT / "extracted_images"
PHASE_ROOT  = ROOT / "phase_annotations"
TOOL_ROOT   = ROOT / "tool_annotations"
OUT_PKL     = KAYOUNG_ROOT / "dataframes/cholec_split_250px_25fps.pkl"
OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

# --------- Label/column maps ---------
PHASE2ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}
TOOL_COL_MAP = {
    "Grasper":      "tool_Grasper",
    "Bipolar":      "tool_Bipolar",
    "Hook":         "tool_Hook",
    "Scissors":     "tool_Scissors",
    "Clipper":      "tool_Clipper",
    "Irrigator":    "tool_Irrigator",
    "SpecimenBag":  "tool_SpecimenBag",
}
TOOL_COLS = list(TOOL_COL_MAP.values())

# --------- Filename parsing ---------
IMG_NAME_RE = re.compile(r".*?_(\d+)\.png$", re.IGNORECASE)   # videoNN_MMMMMM.png -> MMMMMM
VID_NAME_RE = re.compile(r"video\s*0*([0-9]+)$", re.IGNORECASE)  # videoNN -> NN

def parse_video_idx(dirname: str) -> int:
    m = VID_NAME_RE.search(dirname)
    if not m:
        raise ValueError(f"Cannot parse video index from dirname: {dirname}")
    return int(m.group(1))

def list_video_dirs(img_root: Path):
    dirs = [d for d in img_root.iterdir() if d.is_dir()]
    # sort numerically
    dirs = [d for d in dirs if VID_NAME_RE.search(d.name)]
    return sorted(dirs, key=lambda d: parse_video_idx(d.name))

def list_images_with_frames(video_dir: Path):
    """Return PNGs in the video folder, sorted and paired with frame indices."""
    imgs = sorted(p for p in video_dir.glob("*.png"))
    rows = []
    for p in imgs:
        m = IMG_NAME_RE.match(p.name)
        if not m:
            continue
        frame = int(m.group(1))
        rows.append((p, frame))
    return rows

# --------- Label file parsers ---------
def read_phase_file(phase_path: Path) -> pd.DataFrame:
    # Example:
    # Frame   Phase
    # 0       Preparation
    df = pd.read_csv(phase_path, sep=r"\s+|\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = df[["Frame", "Phase"]].copy()
    df["class"] = df["Phase"].map(PHASE2ID)
    df = df[["Frame", "class"]]
    return df

def read_tool_file(tool_path: Path) -> pd.DataFrame:
    # Example:
    # Frame   Grasper Bipolar Hook Scissors Clipper Irrigator SpecimenBag
    df = pd.read_csv(tool_path, sep=r"\s+|\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    # rename to standard column names
    rename = {src: dst for src, dst in TOOL_COL_MAP.items() if src in df.columns}
    df = df.rename(columns=rename)
    # create missing tool columns with 0
    for col in TOOL_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[["Frame"] + TOOL_COLS].copy()
    return df

# --------- Build per-video DataFrame ---------
def build_df_for_video(video_dir: Path, img_root: Path, phase_root: Path, tool_root: Path) -> pd.DataFrame:
    vid_name  = video_dir.name
    video_idx = parse_video_idx(vid_name)

    # image list
    img_list = list_images_with_frames(video_dir)
    if not img_list:
        return pd.DataFrame()

    rel_paths = [p.relative_to(img_root).as_posix() for p, f in img_list]
    frames    = [f for _, f in img_list]
    out = pd.DataFrame({
        "video_idx": video_idx,
        "image_path": rel_paths,
        "index": frames,
    })

    # ----- PHASE -----
    phase_candidates = [
        phase_root / f"{vid_name}-phase.txt",
        phase_root / f"{vid_name}_phase.txt",
        phase_root / f"{vid_name.upper()}-phase.txt",
    ]
    phase_df = None
    for cand in phase_candidates:
        if cand.exists():
            phase_df = read_phase_file(cand)
            break

    if phase_df is not None:
        s = (
            phase_df.set_index("Frame")["class"]
                    .sort_index()
                    .reindex(out["index"])
                    .ffill()     # fill with previous label
                    .bfill()     # for leading gaps, use next label
        )
        out["class"] = s.astype(int).values
    else:
        out["class"] = 0  # if missing, treat all as Preparation(0)

    # ----- TOOL -----
    tool_candidates = [
        tool_root / f"{vid_name}-tool.txt",
        tool_root / f"{vid_name}_tool.txt",
        tool_root / f"{vid_name.upper()}-tool.txt",
    ]
    tool_df = None
    for cand in tool_candidates:
        if cand.exists():
            tool_df = read_tool_file(cand)
            break

    if tool_df is not None:
        tdf = (
            tool_df.set_index("Frame")
                   .sort_index()
                   .reindex(out["index"])
                   .ffill()    # forward fill
                   .fillna(0)  # leading gaps to 0
        )
        tdf = tdf.astype(int).reset_index(drop=True)
        for col in TOOL_COLS:
            out[col] = tdf[col].values
    else:
        for col in TOOL_COLS:
            out[col] = 0

    # sort/reset index
    out = out.sort_values(["video_idx", "index"]).reset_index(drop=True)

    # ensure dtypes
    out["video_idx"] = out["video_idx"].astype(int)
    out["index"]     = out["index"].astype(int)
    out["class"]     = out["class"].astype(int)
    for c in TOOL_COLS:
        out[c] = out[c].astype(int)

    return out

# --------- Build full DataFrame ---------
def build_all_df(img_root: Path, phase_root: Path, tool_root: Path) -> pd.DataFrame:
    rows = []
    for vdir in tqdm(list_video_dirs(img_root), desc="Building DataFrame"):
        dfv = build_df_for_video(vdir, img_root, phase_root, tool_root)
        if len(dfv):
            rows.append(dfv)
    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)

    # column order (match what the loader expects)
    wanted = ["video_idx", "image_path", "index", "class"] + TOOL_COLS
    df_all = df_all[wanted]

    # NaN/domain checks (for training code asserts)
    assert df_all[wanted].isnull().sum().sum() == 0, "NaN detected in dataframe!"
    assert set(df_all["class"].unique()) <= set(range(7)), "class must be in [0..6]"
    for c in TOOL_COLS:
        vals = set(pd.unique(df_all[c].values.ravel()))
        assert vals <= {0, 1}, f"{c} must be 0/1, but found {vals}"

    return df_all

if __name__ == "__main__":
    df = build_all_df(IMG_ROOT, PHASE_ROOT, TOOL_ROOT)
    print(df.head())
    print(df.tail())
    print(df.shape, df.columns.tolist())
    df.to_pickle(OUT_PKL)
    print(f"[OK] Saved DataFrame to: {OUT_PKL}")
