#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageSequence, ImageDraw, UnidentifiedImageError

# ===== Paths =====
SRC_ROOT = Path("./visualized_neuron_concept_20_with_concept_info_integrity/ASFormer/4_Video_wise_Top1")
DST_ROOT = Path("./visualized_neuron_concept_20_with_concept_info_final/ASFormer/4_Video_wise_Top1")
DST_ROOT.mkdir(parents=True, exist_ok=True)

# ===== Settings =====
DEFAULT_DURATION = 100  # ms
PRINT_SKIPPED = True

# Prefer imageio if available
try:
    import imageio.v3 as iio  # latest
    HAS_IMAGEIO_V3 = True
except Exception:
    HAS_IMAGEIO_V3 = False
try:
    import imageio  # older fallback
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False


# -----------------------------
# Sorting / discovery utilities
# -----------------------------
def natural_key(p: Path):
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_neuron_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("neuron")], key=natural_key)

def list_seq_gifs(neuron_dir: Path) -> List[Path]:
    gifs = [p for p in neuron_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".gif" and p.name.startswith("seq")]
    return sorted(gifs, key=natural_key)


# -----------------------------
# Strict Pillow-based restorer (fallback)
# -----------------------------
def _frame_bbox(im: Image.Image) -> Tuple[int, int, int, int]:
    bbox = getattr(im, "dispose_extent", None)
    if bbox and isinstance(bbox, tuple) and len(bbox) == 4:
        return bbox
    try:
        if im.tile and len(im.tile) > 0:
            _, region, _ = im.tile[0][:3]
            if region and len(region) >= 4:
                return region[:4]
    except Exception:
        pass
    return (0, 0, im.width, im.height)

def _palette_index_to_rgba(im: Image.Image, idx: int) -> Tuple[int, int, int, int]:
    pal = im.getpalette()
    if pal is None:
        return (0, 0, 0, 255)
    base = 3 * idx
    r = pal[base + 0] if base + 0 < len(pal) else 0
    g = pal[base + 1] if base + 1 < len(pal) else 0
    b = pal[base + 2] if base + 2 < len(pal) else 0
    return (r, g, b, 255)

def _gif_background_rgba(im: Image.Image) -> Tuple[int, int, int, int]:
    # If a GIF background index exists, use that palette color; otherwise opaque black
    bg_idx = im.info.get("background", None)
    if isinstance(bg_idx, int):
        try:
            return _palette_index_to_rgba(im, bg_idx)
        except Exception:
            pass
    return (0, 0, 0, 255)

def decode_gif_full_frames_pillow(gif_path: Path) -> Tuple[List[Image.Image], List[int]]:
    """
    Reconstruct full-canvas RGBA frames using only Pillow.
    Handles disposal 2/3, transparency, and regional offsets.
    """
    frames: List[Image.Image] = []
    durations: List[int] = []

    with Image.open(gif_path) as im:
        canvas_w, canvas_h = im.size
        bg_rgba = _gif_background_rgba(im)

        accum = Image.new("RGBA", (canvas_w, canvas_h), bg_rgba)
        accum_prev = None

        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            # Offset/region
            x0, y0, x1, y1 = _frame_bbox(im)
            # Current frame RGBA (palette + transparency applied)
            cur = im.convert("RGBA")

            # Prepare for disposal=3
            accum_prev = accum.copy()

            # Composite at the exact region
            accum.paste(cur, (x0, y0), cur)

            # Capture full frame at this moment
            frames.append(accum.copy())

            # Duration
            dur = im.info.get("duration", DEFAULT_DURATION)
            durations.append(int(dur) if dur and dur > 0 else DEFAULT_DURATION)

            # Disposal handling
            disposal = getattr(im, "disposal_method", None)
            if disposal is None:
                disposal = getattr(im, "disposal", 0)
            if disposal == 2:
                draw = ImageDraw.Draw(accum)
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=bg_rgba)
                del draw
            elif disposal == 3 and accum_prev is not None:
                accum = accum_prev

    return frames, durations


# -----------------------------
# Accurate imageio-based restorer (preferred)
# -----------------------------
def decode_gif_full_frames_imageio(gif_path: Path) -> Tuple[List[Image.Image], List[int]]:
    """
    Use imageio to return fully-restored frames (disposal/offset/transparency handled).
    Prefers v3; falls back to v2.
    """
    frames: List[Image.Image] = []
    durations: List[int] = []

    if HAS_IMAGEIO_V3:
        arr = iio.imread(gif_path, index=None)  # (n, H, W, 4 or 3)
        meta = iio.immeta(gif_path) or {}
        # v3 metadata may not include per-frame durations; use a common value
        dur = int(meta.get("duration", DEFAULT_DURATION)) or DEFAULT_DURATION
        for a in arr:
            # Ensure RGBA
            if a.shape[-1] == 3:
                from numpy import dstack, uint8, ones
                a = dstack([a, 255 * ones(a[..., :1].shape, dtype=uint8)])
            frames.append(Image.fromarray(a, "RGBA"))
            durations.append(dur)
        return frames, durations

    if HAS_IMAGEIO:
        reader = imageio.get_reader(str(gif_path))
        try:
            meta = reader.get_meta_data() or {}
            dur = int(meta.get("duration", DEFAULT_DURATION)) or DEFAULT_DURATION
        except Exception:
            dur = DEFAULT_DURATION
        try:
            for im in reader:
                pil = Image.fromarray(im)  # imageio already applied disposal to full frame
                if pil.mode != "RGBA":
                    pil = pil.convert("RGBA")
                frames.append(pil)
                durations.append(dur)
        finally:
            reader.close()
        return frames, durations

    # If imageio is not available, fall back to Pillow
    return decode_gif_full_frames_pillow(gif_path)

# ... (imports and utilities above remain the same) ...

# -----------------------------
# Main: restore sequences → concatenate → save
# -----------------------------
def concat_sequences_after_full_restore(neuron_dir: Path, out_path: Path):
    seq_paths = list_seq_gifs(neuron_dir)
    if not seq_paths:
        if PRINT_SKIPPED:
            print(f"[SKIP] {neuron_dir.name}: no seq*.gif found")
        return

    # First, restore each sequence into full-canvas RGBA frames
    restored_frames_all: List[Image.Image] = []
    durations_all: List[int] = []

    iterable = seq_paths
    if use_tqdm:
        iterable = tqdm(seq_paths, desc=f"{neuron_dir.name}", leave=False)

    canvas_w = canvas_h = None
    for gif_path in iterable:
        try:
            frames, durs = decode_gif_full_frames_imageio(gif_path)
        except Exception:
            # If imageio fails, retry with Pillow
            frames, durs = decode_gif_full_frames_pillow(gif_path)

        if not frames:
            print(f"[WARN] failed to restore frames: {gif_path}")
            continue

        # Update canvas size (max across inputs)
        w, h = frames[0].size
        if canvas_w is None:
            canvas_w, canvas_h = w, h
        else:
            canvas_w = max(canvas_w, w)
            canvas_h = max(canvas_h, h)

        restored_frames_all.extend(frames)
        durations_all.extend(durs)

    if not restored_frames_all:
        if PRINT_SKIPPED:
            print(f"[SKIP] {neuron_dir.name}: no restored frames")
        return

    # ===== Key modification (stronger transparency removal) =====
    target_size = (canvas_w, canvas_h)
    final_frames_rgb = []  # final frames to save (RGB mode)

    # Composite every restored frame over an opaque black canvas and drop alpha
    for fr in restored_frames_all:
        # 1) Create an opaque black RGBA canvas
        black_background_canvas = Image.new("RGBA", target_size, (0, 0, 0, 255))

        # 2) Paste original frame on top using its alpha as mask
        #    If sizes differ and centering is desired, compute offsets.
        #    For now, paste at (0, 0).
        black_background_canvas.paste(fr, (0, 0), mask=fr)

        # 3) Convert to RGB to remove alpha
        final_frames_rgb.append(black_background_canvas.convert("RGB"))
    # ============================================================

    # Save: disposal=1 (Do not dispose), optimize=False — safe since frames are full-canvas
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = final_frames_rgb[0], final_frames_rgb[1:]

    # Provide background color explicitly to Pillow when saving the GIF.
    # With RGB frames, specifying an RGB tuple is often safest; if it errors, fallback to 0.
    save_options = {
        "save_all": True,
        "append_images": rest,
        "duration": durations_all,
        "loop": 0,
        "optimize": False,  # safe to keep False with full frames
        "disposal": 1,
        "background": (0, 0, 0),  # explicitly mark black as the background in the GIF header
    }

    try:
        first.save(out_path, **save_options)
    except TypeError as e:
        print(f"[WARN] Pillow background=(0,0,0) error: {e}. Retrying with background=0.")
        save_options["background"] = 0  # palette index 0 as background (typically black)
        first.save(out_path, **save_options)
    
    print(f"[OK]  {neuron_dir.name} -> {out_path}")

def main():
    if not SRC_ROOT.exists():
        print(f"[ERROR] Input path not found: {SRC_ROOT.resolve()}")
        return
    neuron_dirs = list_neuron_dirs(SRC_ROOT)
    if not neuron_dirs:
        print(f"[ERROR] No neuron folders found under: {SRC_ROOT}")
        return

    iterable = neuron_dirs
    if use_tqdm:
        iterable = tqdm(neuron_dirs, desc="Restoring & concatenating")

    for nd in iterable:
        out_path = DST_ROOT / f"{nd.name}.gif"
        concat_sequences_after_full_restore(nd, out_path)

if __name__ == "__main__":
    main()
