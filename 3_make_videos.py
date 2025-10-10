import os
import re
import glob
from typing import List, Tuple
from PIL import Image, ImageOps, ImageSequence
from tqdm import tqdm


spr_model = "ASFormer"
selection_method = "4_Video_wise_Top1"
concept_sets = "ChoLec-270"
neuron_concepts = f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"


ROWS, COLS = 4, 5
CELLS_PER_GRID = ROWS * COLS
PAD = 0
DURATION_MS = 200

SRC_BASE = f"visualized_neuron_concept/{spr_model}/{selection_method}"
DST_BASE = f"visualized_neuron_concept_{ROWS*COLS}/{spr_model}/{selection_method}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_seq_num(path: str) -> int:
    m = re.search(r"seq(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def to_rgb_on_black(im: Image.Image) -> Image.Image:
    """Composite transparent/semi-transparent images on a black background (handle alpha only; keep colors)."""
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info) or (im.mode == "P"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        composited = Image.alpha_composite(bg, rgba)
        return composited.convert("RGB")
    return im.convert("RGB")


def load_gif_frames(path: str) -> Tuple[List[Image.Image], int]:
    """Load a GIF into a list of RGB frames and return an estimated representative duration."""
    try:
        im = Image.open(path)
    except Exception:
        return [], DURATION_MS

    frames, durations = [], []
    for fr in ImageSequence.Iterator(im):
        rgb = to_rgb_on_black(fr)  # replace transparency with black
        frames.append(rgb.copy())
        d = fr.info.get("duration", 0) or 0
        durations.append(d)

    if not frames:
        return [], DURATION_MS

    valid = [d for d in durations if d and d > 0]
    duration = int(round(sum(valid) / len(valid))) if valid else DURATION_MS
    return frames, duration


def pad_without_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """No scaling: place the image centered on a black canvas of target_size."""
    W, H = target_size
    bg = Image.new("RGB", (W, H), (0, 0, 0))
    x = (W - img.width) // 2
    y = (H - img.height) // 2
    bg.paste(img, (x, y))
    return bg


def make_global_palette(frames: List[Image.Image], max_samples: int = 24) -> Image.Image:
    """
    Build a global palette by stacking a subset of frames on a sheet and
    converting to an adaptive 256-color 'P' image. Returns the palette carrier.
    """
    if not frames:
        # Safety: avoid empty palette
        tmp = Image.new("RGB", (16, 16), (0, 0, 0))
        return tmp.convert("P", palette=Image.ADAPTIVE, colors=256)

    n = min(len(frames), max_samples)
    # Uniform sampling across frames
    idxs = [round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)]
    samples = [frames[i] for i in idxs]

    w, h = samples[0].size
    sheet = Image.new("RGB", (w, h * n))
    for i, f in enumerate(samples):
        sheet.paste(f, (0, i * h))

    pal_img = sheet.convert("P", palette=Image.ADAPTIVE, colors=256)

    # Force palette index 0 to black (0,0,0)
    pal = pal_img.getpalette()
    pal[0:3] = [0, 0, 0]
    pal_img.putpalette(pal)

    # Remove any transparency metadata
    pal_img.info.pop("transparency", None)
    return pal_img


def build_grid_animation(gif_paths: List[str], out_path: str):
    """
    Combine up to ROWS×COLS GIFs into a grid animation and save as a GIF.
      - No resizing (loss-averse): cell size is the max (W, H) across inputs
      - Missing cells are filled with black frames
      - Use a single 256-color global palette without dithering
    """
    ensure_dir(os.path.dirname(out_path))

    # Load input GIFs
    loaded: List[List[Image.Image]] = []
    durations = []
    for p in gif_paths:
        frames, dur = load_gif_frames(p)
        loaded.append(frames)
        durations.append(dur)

    # Cell size = max (W, H) among inputs → padding only, no scaling
    max_w, max_h = 0, 0
    for frames in loaded:
        if frames:
            w, h = frames[0].size  # typically consistent within a GIF
            max_w = max(max_w, w)
            max_h = max(max_h, h)
    if max_w == 0 or max_h == 0:
        max_w, max_h = 640, 360  # fallback

    cell_size = (max_w, max_h)

    # Apply padding only (no resizing)
    padded_sets: List[List[Image.Image]] = []
    for frames in loaded:
        if not frames:
            padded_sets.append([])
            continue
        ps = [pad_without_resize(f, cell_size) for f in frames]
        padded_sets.append(ps)

    while len(padded_sets) < CELLS_PER_GRID:
        padded_sets.append([])

    max_T = max((len(ps) for ps in padded_sets), default=1) or 1

    cell_w, cell_h = cell_size
    W = COLS * cell_w + (COLS - 1) * PAD
    H = ROWS * cell_h + (ROWS - 1) * PAD
    black_cell = Image.new("RGB", cell_size, (0, 0, 0))
    frames_out: List[Image.Image] = []

    for t in range(max_T):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        for idx in range(CELLS_PER_GRID):
            r, c = divmod(idx, COLS)
            x = c * (cell_w + PAD)
            y = r * (cell_h + PAD)
            ps = padded_sets[idx]
            fr = ps[t % len(ps)] if ps else black_cell
            canvas.paste(fr, (x, y))
        frames_out.append(canvas)

    # Global palette for better cross-frame color consistency; disable dithering
    pal_img = make_global_palette(frames_out, max_samples=24)

    # Quantize each frame with the global palette and remove transparency metadata
    quantized = []
    for im in frames_out:
        q = im.quantize(palette=pal_img, dither=Image.Dither.NONE)
        q.info.pop("transparency", None)
        quantized.append(q)

    # Choose representative duration: maximum valid input duration (aiming to preserve original timing)
    duration_out = max([d for d in durations if d and d > 0], default=DURATION_MS)

    # Save GIF with conservative settings
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_out,
        loop=0,
        optimize=False,
        disposal=2,
        background=0,  # background index = 0 (black)
        # do not set "transparency"
    )

def main():
    ensure_dir(DST_BASE)
    neuron_dirs = sorted(glob.glob(os.path.join(SRC_BASE, "neuron*")))
    if not neuron_dirs:
        print(f"[WARN] No neuron folders under: {SRC_BASE}")
        return

    for ndir in tqdm(neuron_dirs, desc="Neurons", dynamic_ncols=True):
        gifs = sorted(glob.glob(os.path.join(ndir, "*.gif")), key=parse_seq_num)
        if not gifs:
            continue

        rel = os.path.relpath(ndir, SRC_BASE)  # neuronXXX
        out_neuron_dir = os.path.join(DST_BASE, rel)
        ensure_dir(out_neuron_dir)

        for i in range(0, len(gifs), CELLS_PER_GRID):
            chunk = gifs[i:i + CELLS_PER_GRID]
            start_num = parse_seq_num(chunk[0]) if chunk else i
            end_num = parse_seq_num(chunk[-1]) if chunk else (i + len(chunk) - 1)
            out_name = f"seq{start_num:03d}-{end_num:03d}.gif"
            out_path = os.path.join(out_neuron_dir, out_name)
            build_grid_animation(chunk, out_path)

    print(f"[OK] Saved 3x4 grid GIFs under: {DST_BASE}")


if __name__ == "__main__":
    main()
