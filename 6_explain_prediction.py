#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed  # parallel workers

# ====== Settings ======
CONTRIB_PKL = Path("spr_models/ASFormer/ASFormer_test_contributions.pkl")
IMAGE_ROOT  = Path("/data2/local_datasets/cholec80/cholec_split_360x640_1fps")
NEURON_JSON = Path("extracted_neuron_concepts/ASFormer/4_Video_wise_Top1_and_ChoLec-270.json")
OUT_ROOT    = Path("/data2/local_datasets/kayoung_data/explanation")   # output root

IMAGE_EXT   = ".png"
VIDEO_NUMBER_START = 41

# ====== Styles ======
PANEL_BG = (25, 25, 28, 210)  # translucent dark panel
CANVAS_BG = (8, 8, 10)        # overall background
TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (205, 205, 210)
ACCENT = (160, 160, 255)

PADDING = 16
GRID_ROWS, GRID_COLS = 1, 1
TILE_MARGIN = 0  # not used (single image)
TITLE_SIZE = 22
HEADER_SIZE = 20
BODY_SIZE = 18
CONCEPT_SIZE = 16      # font size for concepts
LINE_GAP = 6
SECTION_GAP = 10
PANEL_RADIUS = 16
BULLET_RADIUS = 5
PANEL_INNER_PAD = 14
EXTRA_BOTTOM = 600  # extra space (e.g., 400px)

# Concept color palette (extend if needed)
CONCEPT_PALETTE = [
    (255, 180, 180), (255, 220, 165), (255, 250, 170),
    (200, 245, 180), (175, 230, 255), (195, 190, 255),
    (245, 190, 255), (255, 205, 230), (170, 235, 210),
    (210, 210, 255), (255, 205, 180), (200, 255, 210),
]

# ====== Cholec80 phases (0..6) ======
PHASE_NAMES = [
    "Preparation",                 # 0 -> 1
    "Calot Triangle Dissection",   # 1 -> 2
    "Clipping and Cutting",        # 2 -> 3
    "Gallbladder Dissection",      # 3 -> 4
    "Gallbladder Packaging",       # 4 -> 5
    "Cleaning Coagulation",        # 5 -> 6
    "Gallbladder Retraction",      # 6 -> 7
]

def format_phase(c: int) -> str:
    """0-based class idx -> 'n · PhaseName' (n = c+1). If out of range, return the number only."""
    n = c + 1
    if 0 <= c < len(PHASE_NAMES):
        return f"{n} · {PHASE_NAMES[c]}"
    return str(n)

# number of neuron concepts to show
TOP_CONCEPTS = 5

# Debug: None means process all frames per video
MAX_FRAMES_PER_VIDEO: Optional[int] = None


def _worker_render_one_video(
    vid_i: int,
    contribs: torch.Tensor,
    preds: torch.Tensor,
    gts: torch.Tensor,
    neuron_map: Dict[int, Dict[str, Any]],
    image_root: Path,
    out_root: Path,
    image_ext: str,
):
    """
    Render one video in a worker process.
    Large tensors are passed via pickle; depending on OS this can have some overhead,
    but splitting by video is usually fast enough.
    """
    try:
        # Reduce CPU contention (limit threads used by each worker)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        visualize_video(
            video_idx=vid_i,
            contribs=contribs,
            preds=preds,
            gts=gts,
            neuron_map=neuron_map,
            image_root=image_root,
            out_root=out_root,
            image_ext=image_ext,
        )
        return (vid_i, True, "")
    except Exception as e:
        return (vid_i, False, str(e))

def _textsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _wrap_text_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap the text so that each line width does not exceed max_width."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        candidate = cur + " " + w
        if _textsize(draw, candidate, font)[0] <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_rounded_panel(canvas: Image.Image, xy, radius: int, fill_rgba):
    """Draw a translucent rounded rectangle by compositing (Pillow's rounded_rectangle blends alpha weakly)."""
    from PIL import ImageDraw, Image
    x0, y0, x1, y1 = xy
    panel_w, panel_h = x1 - x0, y1 - y0
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle((0, 0, panel_w, panel_h), radius=radius, fill=fill_rgba)
    canvas.alpha_composite(panel, (x0, y0))


def _concept_color(name: str) -> tuple:
    """Stable hash of name -> palette color."""
    idx = (hash(name) & 0xFFFFFFFF) % len(CONCEPT_PALETTE)
    return CONCEPT_PALETTE[idx]


def load_font():
    candidates = [
        ("./Pretendard-Black.otf", TITLE_SIZE),
        ("./Pretendard-Medium.otf", HEADER_SIZE),
        ("./Pretendard-Regular.otf", CONCEPT_SIZE),
        ("./Pretendard-Medium.otf", CONCEPT_SIZE),   # concept font fallback
        ("./Pretendard-ExtraLight.otf", CONCEPT_SIZE),
    ]
    font_title = font_header = font_body = font_concept = ImageFont.load_default()
    for path, size in candidates:
        if Path(path).exists():
            try:
                f = ImageFont.truetype(path, size)
                if size == TITLE_SIZE:
                    font_title = f
                elif size == HEADER_SIZE:
                    font_header = f
                elif size == CONCEPT_SIZE:
                    font_concept = f
                else:
                    if font_body == ImageFont.load_default():
                        font_body = f
            except Exception:
                continue
    if font_header == ImageFont.load_default():
        font_header = font_body
    if font_concept == ImageFont.load_default():
        font_concept = font_body   # fallback
    return font_title, font_header, font_body, font_concept


def load_contributions(pkl_path: Path):
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_neuron_json(json_path: Path) -> Dict[int, Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    index_map: Dict[int, Dict[str, Any]] = {}
    if isinstance(js, dict):
        if "neuron_idx" in js:
            index_map[js["neuron_idx"]] = js
        else:
            for v in js.values():
                if isinstance(v, dict) and "neuron_idx" in v:
                    index_map[v["neuron_idx"]] = v
    elif isinstance(js, list):
        for item in js:
            if isinstance(item, dict) and "neuron_idx" in item:
                index_map[item["neuron_idx"]] = item
    return index_map


def frame_to_filename_num(t: int) -> int:
    return 1 + 25 * t


def open_image_or_blank(path: Path, size_like: Optional[Image.Image]) -> Image.Image:
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    if size_like is not None:
        return Image.new("RGB", size_like.size, (30, 30, 30))
    return Image.new("RGB", (640, 360), (30, 30, 30))

def visualize_video(
    video_idx: int,
    contribs: torch.Tensor,  # (T, 64, C)
    preds: torch.Tensor,     # (T,)
    gts: torch.Tensor,       # (T,)
    neuron_map: Dict[int, Dict[str, Any]],
    image_root: Path,
    out_root: Path,
    image_ext: str = ".png",
):
    T, num_neurons, num_classes = contribs.shape
    video_num = VIDEO_NUMBER_START + video_idx
    video_name = f"video{video_num:02d}" if video_num < 100 else f"video{video_num}"
    video_dir = image_root / video_name

    # Sample size
    sample_num = frame_to_filename_num(0)
    sample_path = video_dir / f"{video_name}_{sample_num:06d}{image_ext}"
    sample_img = open_image_or_blank(sample_path, None).convert("RGBA")
    tile_w, tile_h = sample_img.size

    font_title, font_header, font_body, font_concept = load_font()

    # Limit frames per video if requested
    frame_range = range(T)
    if MAX_FRAMES_PER_VIDEO is not None:
        frame_range = range(min(T, MAX_FRAMES_PER_VIDEO))

    out_dir = out_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Side panel layout parameters =====
    PANEL_WIDTH = max(int(tile_w * 0.55), 520)  # panel width relative to image
    GUTTER = 16                                  # gap between image and panel
    canvas_w = tile_w + GUTTER + PANEL_WIDTH
    canvas_bg_rgba = CANVAS_BG + (255,)

    for t in frame_range:
        base_num = frame_to_filename_num(t)
        img_path = video_dir / f"{video_name}_{base_num:06d}{image_ext}"
        img = open_image_or_blank(img_path, sample_img).convert("RGBA")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.BILINEAR)

        # -------- Prepare data --------
        pred_c = int(preds[t].item())
        gt_c = int(gts[t].item())
        contrib_vec = contribs[t, :, pred_c]  # (64,)

        max_val = float(torch.max(contrib_vec).item())
        thr = 0.9 * max_val
        cand_idx = (contrib_vec >= thr).nonzero(as_tuple=True)[0].tolist()
        if not cand_idx:
            cand_idx = [int(torch.argmax(contrib_vec).item())]
        cand_idx = sorted(cand_idx, key=lambda i: float(contrib_vec[i].item()), reverse=True)

        # -------- Measure wrapped text height --------
        tmp = Image.new("RGBA", (10, 10))
        td = ImageDraw.Draw(tmp)
        max_text_w = PANEL_WIDTH - 2 * (PADDING + PANEL_INNER_PAD)

        title_line = f"{video_name}  ·  t={t}  ·  {video_name}_{base_num:06d}{image_ext}"
        title_h = _textsize(td, title_line, font_title)[1]

        # -------- Info --------
        if pred_c == gt_c:
            pred_fill = (0, 180, 0)   # green
            gt_fill   = (0, 180, 0)
            font_info = font_header   # bold-ish font
        else:
            pred_fill = (200, 0, 0)   # red
            gt_fill   = (200, 0, 0)
            font_info = font_header

        info_pairs = [
            (f"Predicted: {format_phase(pred_c)}", pred_fill, font_info),
            (f"GT: {format_phase(gt_c)}", gt_fill, font_info),
        ]

        # Height of info block
        info_h = 0
        for text, fill, font_used in info_pairs:
            wrapped = _wrap_text_by_width(td, text, font_used, max_text_w)
            for wl in wrapped:
                info_h += _textsize(td, wl, font_used)[1] + LINE_GAP
        if info_h:
            info_h -= LINE_GAP

        # Neuron blocks height
        neuron_h = 0
        measured_neurons: List[Dict[str, Any]] = []
        for ni in cand_idx:
            n_header = f"Neuron {ni}  (contrib={float(contrib_vec[ni].item()):.6f})"
            h_h = _textsize(td, n_header, font_header)[1] + LINE_GAP

            neuron_info = neuron_map.get(int(ni))
            concepts = []
            if neuron_info and isinstance(neuron_info.get('selected_concepts_sorted'), list):
                sel = neuron_info['selected_concepts_sorted'][:TOP_CONCEPTS]
                concepts = ["(No concepts listed)"] if not sel else [
                    (str(itm.get("name","")), itm.get("avg_cos_sim", None)) for itm in sel
                ]
            else:
                concepts = [("(No concept metadata found)", None)]

            # Measure wrapped concept lines
            concept_wrapped: List[List[str]] = []
            concept_names_for_color: List[str] = []
            ch_h = 0
            for name, score in concepts:
                label = name if score is None else f"{name} ({score:.4f})"
                wrapped = _wrap_text_by_width(td, label, font_concept, max_text_w - 2*(BULLET_RADIUS + 6))

                concept_wrapped.append(wrapped)
                concept_names_for_color.append(name)
                if wrapped:
                    ch_h += max(_textsize(td, wrapped[0], font_concept)[1], 2*BULLET_RADIUS + 6) + LINE_GAP
                    for cont in wrapped[1:]:
                        ch_h += _textsize(td, cont, font_concept)[1] + LINE_GAP
            block_h = h_h + ch_h + SECTION_GAP
            neuron_h += block_h

            measured_neurons.append({
                "ni": ni,
                "header": n_header,
                "concept_wrapped": concept_wrapped,
                "concept_names_for_color": concept_names_for_color,
            })
        if neuron_h:
            neuron_h -= SECTION_GAP

        # Panel height equals image height by default; if content is larger, extend canvas height
        content_h = (PANEL_INNER_PAD + title_h + SECTION_GAP +
                     info_h + SECTION_GAP + neuron_h + PANEL_INNER_PAD)
        canvas_h = max(tile_h + 700, content_h)  # ensure panel area is at least image height
        canvas = Image.new("RGBA", (canvas_w, canvas_h), canvas_bg_rgba)

        # ---- Left: image ----
        # vertically center the image
        img_y = (canvas_h - tile_h) // 2
        canvas.alpha_composite(img, (0, img_y))

        # ---- Right: panel ----
        panel_x0 = tile_w + GUTTER
        panel_y0 = 0
        panel_x1 = panel_x0 + PANEL_WIDTH
        panel_y1 = canvas_h
        _draw_rounded_panel(
            canvas,
            (panel_x0 + PADDING, panel_y0 + PADDING, panel_x1 - PADDING, panel_y1 - PADDING),
            PANEL_RADIUS,
            PANEL_BG
        )

        draw = ImageDraw.Draw(canvas)
        cur_x = panel_x0 + PADDING + PANEL_INNER_PAD
        cur_y = panel_y0 + PADDING + PANEL_INNER_PAD

        # Title
        draw.text((cur_x, cur_y), title_line, font=font_title, fill=TEXT_PRIMARY)
        cur_y += title_h + SECTION_GAP

        # Info
        for text, fill, font_used in info_pairs:
            wrapped = _wrap_text_by_width(draw, text, font_used, max_text_w)
            for wl in wrapped:
                draw.text((cur_x, cur_y), wl, font=font_used, fill=fill)
                cur_y += _textsize(draw, wl, font_used)[1] + LINE_GAP
        cur_y += SECTION_GAP

        # Neurons
        for block in measured_neurons:
            draw.text((cur_x, cur_y), block["header"], font=font_header, fill=ACCENT)
            cur_y += _textsize(draw, block["header"], font_header)[1] + LINE_GAP

            for name_for_color, wrapped in zip(block["concept_names_for_color"], block["concept_wrapped"]):
                bcol = _concept_color(name_for_color)
                # bullet
                draw.ellipse(
                    (cur_x, cur_y + 6, cur_x + 2*BULLET_RADIUS, cur_y + 6 + 2*BULLET_RADIUS),
                    fill=bcol, outline=None
                )
                # first line
                first = wrapped[0] if wrapped else ""
                draw.text((cur_x + 2*BULLET_RADIUS + 8, cur_y), first, font=font_concept, fill=TEXT_PRIMARY)
                line_h = max(_textsize(draw, first, font_body)[1], 2*BULLET_RADIUS + 6)
                cur_y += line_h + LINE_GAP
                # continuation lines
                for cont in wrapped[1:]:
                    draw.text((cur_x + 2*BULLET_RADIUS + 8, cur_y), cont, font=font_concept, fill=TEXT_PRIMARY)
                    cur_y += _textsize(draw, cont, font_body)[1] + LINE_GAP

            cur_y += SECTION_GAP

        # Save
        out_path = out_dir / f"{video_name}_{base_num:06d}.png"
        canvas.convert("RGB").save(out_path, format="PNG")


def main():
    contrib_data = load_contributions(CONTRIB_PKL)
    neuron_map = load_neuron_json(NEURON_JSON)

    # Build task list per video
    tasks = []
    for vid_i, trio in enumerate(contrib_data):
        if not isinstance(trio, (list, tuple)) or len(trio) != 3:
            print(f"[WARN] Unexpected item at index {vid_i}, skipping.")
            continue

        contribs, preds, gts = trio
        if not (isinstance(contribs, torch.Tensor) and contribs.ndim == 3):
            print(f"[WARN] contribs invalid for video index {vid_i}, skipping.")
            continue
        if not (isinstance(preds, torch.Tensor) and preds.ndim == 1):
            print(f"[WARN] preds invalid for video index {vid_i}, skipping.")
            continue
        if not (isinstance(gts, torch.Tensor) and gts.ndim == 1):
            print(f"[WARN] gts invalid for video index {vid_i}, skipping.")
            continue

        tasks.append((vid_i, contribs, preds, gts))

    if not tasks:
        print("No valid videos to process.")
        return

    print(f"Parallel rendering: {len(tasks)} videos with {N_WORKERS} workers")

    # Run in parallel
    futures = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=None) as ex:
        for (vid_i, contribs, preds, gts) in tasks:
            fut = ex.submit(
                _worker_render_one_video,
                vid_i, contribs, preds, gts,
                neuron_map,
                IMAGE_ROOT,
                OUT_ROOT,
                IMAGE_EXT,
            )
            futures.append(fut)

        # Monitor completion/errors
        done_cnt = 0
        for fut in as_completed(futures):
            vid_i, ok, msg = fut.result()
            done_cnt += 1
            if ok:
                print(f"[{done_cnt}/{len(tasks)}] video index {vid_i} OK")
            else:
                print(f"[{done_cnt}/{len(tasks)}] video index {vid_i} ERROR: {msg}")

    print("Done. Results saved under:", OUT_ROOT.resolve())


if __name__ == "__main__":
    # macOS/Windows compatibility (avoid fork issues on macOS)
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
