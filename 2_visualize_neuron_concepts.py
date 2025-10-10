import os
import glob
import pickle
from typing import List, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm
import json

# =============== Use existing tuple_to_paths as-is ===============
def tuple_to_paths(vid: int, frame_1fps: int) -> List[str]:
    vid += 1
    base = frame_1fps * 25 + 1
    paths = []
    for i in range(50):
        fnum = base - i * 25
        if fnum <= 0:
            break
        p = f"/data2/local_datasets/cholec80/cholec_split_360x640_1fps/video{vid:02d}/video{vid:02d}_{fnum:06d}.png"
        if os.path.exists(p):
            paths.append(p)
    return paths  # newest first


# =============== New visualization utilities (GIF creation) ===============
def load_image_safe(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def make_gif_sequence(paths: List[str],
                      out_gif: str,
                      num_frames: int = 10,
                      target_size: Tuple[int, int] = None,
                      duration_ms: int = 200):
    """
    Args:
        paths: list with newest frames first (output of tuple_to_paths)
        out_gif: output GIF path
        num_frames: total frames in the GIF (pad with black if fewer)
        target_size: unify all frames to this (W, H). If None, infer from first valid frame;
                     if none found, defaults to (640, 360).
        duration_ms: per-frame duration in milliseconds

    Behavior:
        - GIF plays from oldest → newest (so reverse the input which is newest-first)
        - If there are fewer than num_frames frames, pad the beginning with black frames
        - Maintain aspect ratio via letterboxing to fit target_size
    """
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)

    # Determine reference size
    ref_img = None
    for p in paths:
        try:
            ref_img = load_image_safe(p)
            break
        except Exception:
            continue
    if target_size is None:
        if ref_img is not None:
            target_size = ref_img.size  # (W, H)
        else:
            target_size = (640, 360)

    W, H = target_size
    black = Image.new("RGB", (W, H), (0, 0, 0))

    # Order from oldest → newest (paths is newest-first, so reverse)
    ordered = list(reversed(paths[:num_frames]))

    # Number of black pads at the beginning
    pad_count = num_frames - len(ordered)
    frames = []

    # Leading padding
    for _ in range(max(0, pad_count)):
        frames.append(black.copy())

    # Actual frames
    for p in ordered:
        try:
            img = load_image_safe(p)
            # Keep aspect ratio and letterbox (contain)
            img_contained = ImageOps.contain(img, (W, H), method=Image.Resampling.LANCZOS)
            bg = black.copy()
            bg.paste(img_contained, ((W - img_contained.width) // 2,
                                     (H - img_contained.height) // 2))
            frames.append(bg)
        except Exception:
            frames.append(black.copy())

    # Ensure exactly num_frames
    frames = frames[:num_frames]

    if not frames:
        return

    # Save GIF (infinite loop)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


# =============== Main routine (save GIFs) ===============
def process_one_pkl(input_pkl: str, neuron_concept: List, output_dir: str):
    """
    Args:
        input_pkl: representative_frames *.pkl
            Structure: [ [ (vid, frame, val), ... ], [ ... ], ... ]  (list per neuron)
        neuron_concept: (unused here) selected concept info per neuron
        output_dir: base output directory

    Output layout:
        visualized_neuron_concept/{pkl_basename}/neuronXXX/seqYYY.gif
    """
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    base = os.path.splitext(os.path.basename(input_pkl))[0]
    out_dir = os.path.join(output_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    for n_idx, neuron in enumerate(tqdm(data, desc=f"[{os.path.basename(input_pkl)}] Neurons", dynamic_ncols=True)):
        for s_idx, tup in enumerate(tqdm(neuron, desc="  Seqs", leave=False, dynamic_ncols=True)):
            if not isinstance(tup, (list, tuple)) or len(tup) < 2:
                continue
            vid, frame_1fps = int(tup[0]), int(tup[1])
            paths = tuple_to_paths(vid, frame_1fps)

            # Output path (GIF)
            out_gif = os.path.join(out_dir, f"neuron{n_idx:03d}", f"seq{s_idx:03d}.gif")

            # Always save a 10-frame GIF; pad missing frames with black
            make_gif_sequence(paths, out_gif, num_frames=10, target_size=None, duration_ms=10)

    print(f"[OK] GIF sequences saved under: {out_dir}")


def main():
    spr_model = "ASFormer"
    selection_method = "4_Video_wise_Top1"
    concept_sets = "ChoLec-270"
    neuron_concepts = f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"
    
    input_dir  = f"spr_models/{spr_model}/representative_frames"
    output_dir = f"visualized_neuron_concept/{spr_model}"
    os.makedirs(output_dir, exist_ok=True)

    with open(neuron_concepts, "r", encoding="utf-8") as f:
        neuron_concepts = json.load(f)

    pkl_files = glob.glob(os.path.join(input_dir, f"{selection_method}.pkl"))
    for idx, pkl_path in enumerate(sorted(pkl_files), start=0):
        print(f"\n=== [{idx}/{len(pkl_files)}] {os.path.basename(pkl_path)} ===")
        neuron_concept = next(
            nc["selected_concepts_sorted"] 
            for nc in neuron_concepts 
            if isinstance(nc, dict) and nc.get("neuron_idx") == idx
        )
        process_one_pkl(pkl_path, neuron_concept, output_dir)

if __name__ == "__main__":
    main()
