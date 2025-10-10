from pathlib import Path
import re
import imageio.v2 as imageio  # pip install imageio imageio-ffmpeg
from PIL import Image
import numpy as np

SRC_ROOT = Path("explanation")
OUT_ROOT = Path("explanation_video_mp4_3")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FRAME_DURATION_SEC = 0.1   # Display time per frame (seconds); e.g., 0.5 gives an effective 2 fps
CHUNK_SIZE = 300           # Save in chunks of this many frames
OUTPUT_FPS = 30            # Actual file fps (we repeat frames to match dwell time)
CODEC = "libx264"          # H.264

# Extract frame number from filename: {videoName}_{######}.png → ######
num_pat = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

def get_frame_number(p: Path, fallback: int) -> int:
    m = num_pat.search(p.name)
    return int(m.group(1)) if m else fallback

def ensure_even_size(img: Image.Image) -> Image.Image:
    """H.264 prefers even width/height. If odd, crop the last row/column by 1 px."""
    w, h = img.size
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return img

# Compute the repeat count to match the intended dwell time at the chosen OUTPUT_FPS.
# Example: OUTPUT_FPS=30, FRAME_DURATION_SEC=0.5 → repeat=15
REPEAT = max(1, int(round(FRAME_DURATION_SEC * OUTPUT_FPS)))

for video_dir in sorted(SRC_ROOT.glob("video*")):
    if not video_dir.is_dir():
        continue

    frames = sorted(video_dir.glob("*.png"))
    if not frames:
        print(f"[WARN] {video_dir.name}: no PNGs found")
        continue

    total = len(frames)
    print(f"[INFO] {video_dir.name}: {total} frames → writing MP4 in chunks of {CHUNK_SIZE}")

    # Chunk the sequence
    for start_idx in range(0, total, CHUNK_SIZE):
        chunk = frames[start_idx : start_idx + CHUNK_SIZE]
        start_num = get_frame_number(chunk[0], start_idx + 1)
        end_num   = get_frame_number(chunk[-1], start_idx + len(chunk))

        (OUT_ROOT / f"{video_dir.name}").mkdir(parents=True, exist_ok=True)
        out_mp4 = OUT_ROOT / f"{video_dir.name}/{video_dir.name}_{start_num:06d}-{end_num:06d}.mp4"

        # Determine resolution from the first frame (+ enforce even dimensions)
        with Image.open(chunk[0]) as im0:
            im0 = ensure_even_size(im0.convert("RGB"))
            width, height = im0.size

        # ffmpeg-based writer
        # macro_block_size=1 → prevents automatic resizing (we already fixed parity)
        writer = imageio.get_writer(
            out_mp4,
            fps=OUTPUT_FPS,
            codec=CODEC,
            format="ffmpeg",
            macro_block_size=1,
            quality=8,  # 0 (low) to 10 (high); alternatively set bitrate="4M", etc.
        )

        try:
            for idx, p in enumerate(chunk, 1):
                with Image.open(p) as pil_img:
                    pil_img = ensure_even_size(pil_img.convert("RGB"))
                    # Repeat each frame REPEAT times to achieve the intended dwell time
                    frame_np = np.asarray(pil_img)
                    for _ in range(REPEAT):
                        writer.append_data(frame_np)

                if idx % 30 == 0:
                    print(f"[INFO] {video_dir.name}: {idx}/{len(chunk)} frames processed")

        finally:
            writer.close()

        effective_fps = OUTPUT_FPS / REPEAT  # perceived fps (= 1 / FRAME_DURATION_SEC)
        print(
            f"[OK] {out_mp4.name} written "
            f"({len(chunk)} frames, dwell {FRAME_DURATION_SEC}s/frame, effective FPS ≈ {effective_fps:.3f})"
        )

    print(f"[DONE] {video_dir.name} complete\n")
