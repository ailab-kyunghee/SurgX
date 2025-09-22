#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

# ====== 설정 ======
CONTRIB_PKL = Path("spr_models/ASFormer/ASFormer_test_contributions.pkl")
IMAGE_ROOT  = Path("/data2/local_datasets/cholec80/cholec_split_360x640_1fps")
NEURON_JSON = Path("extracted_neuron_concepts/ASFormer/4_Video_wise_Top1_and_ChoLec-270.json")
OUT_ROOT    = Path("explanation")   # 결과 저장 루트

IMAGE_EXT   = ".png"
VIDEO_NUMBER_START = 41

# ====== 스타일 ======
PANEL_BG = (25, 25, 28, 210)  # 반투명 다크 패널
CANVAS_BG = (8, 8, 10)        # 전체 배경
TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (205, 205, 210)
ACCENT = (160, 160, 255)

PADDING = 16
GRID_ROWS, GRID_COLS = 1, 1
TILE_MARGIN = 0  # 1장만 쓰니 의미 없음
TITLE_SIZE = 22
HEADER_SIZE = 20
BODY_SIZE = 18
CONCEPT_SIZE = 16      # ← 컨셉 전용 폰트 크기 (원하는 만큼 키우세요)
LINE_GAP = 6
SECTION_GAP = 10
PANEL_RADIUS = 16
BULLET_RADIUS = 5
PANEL_INNER_PAD = 14
EXTRA_BOTTOM = 600  # 원하는 만큼 추가(예: 400px)

# 콘셉트 색 팔레트(충분히 길게; 필요시 더 추가)
CONCEPT_PALETTE = [
    (255, 180, 180), (255, 220, 165), (255, 250, 170),
    (200, 245, 180), (175, 230, 255), (195, 190, 255),
    (245, 190, 255), (255, 205, 230), (170, 235, 210),
    (210, 210, 255), (255, 205, 180), (200, 255, 210),
]
# ====== Cholec80 phases (0..6) ======
PHASE_NAMES = [
    "Preparation",               # 0 -> 1
    "Calot Triangle Dissection",   # 1 -> 2
    "Clipping and Cutting",           # 2 -> 3
    "Gallbladder Dissection",     # 3 -> 4
    "Gallbladder Packaging",      # 4 -> 5
    "Cleaning Coagulation",       # 5 -> 6
    "Gallbladder Retraction",     # 6 -> 7
]

def format_phase(c: int) -> str:
    """0-based class idx -> 'n · PhaseName' (n = c+1). 범위를 벗어나면 숫자만 반환."""
    n = c + 1
    if 0 <= c < len(PHASE_NAMES):
        return f"{n} · {PHASE_NAMES[c]}"
    return str(n)

# neuron concept 출력 개수
TOP_CONCEPTS = 5

# 디버그용: None이면 전체 프레임
MAX_FRAMES_PER_VIDEO: Optional[int] = None
# ==================


def _textsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _wrap_text_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """단어 단위로 줄바꿈하여 max_width를 넘지 않도록 줄 리스트 반환."""
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
    """반투명 라운드 사각형 (Pillow의 rounded_rectangle은 알파 혼합이 약해서 수동 합성)"""
    from PIL import ImageDraw, Image
    x0, y0, x1, y1 = xy
    panel_w, panel_h = x1 - x0, y1 - y0
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle((0, 0, panel_w, panel_h), radius=radius, fill=fill_rgba)
    canvas.alpha_composite(panel, (x0, y0))


def _concept_color(name: str) -> tuple:
    """이름 -> 팔레트 색상(안정적 해시)"""
    idx = (hash(name) & 0xFFFFFFFF) % len(CONCEPT_PALETTE)
    return CONCEPT_PALETTE[idx]


def load_font():
    candidates = [
        ("./Pretendard-Black.otf", TITLE_SIZE),
        ("./Pretendard-Medium.otf", HEADER_SIZE),
        ("./Pretendard-Regular.otf", CONCEPT_SIZE),
        ("./Pretendard-Medium.otf", CONCEPT_SIZE),   # ← 추가: 컨셉은 Medium 24
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
        font_concept = font_body   # 폴백
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

    # 샘플 크기
    sample_num = frame_to_filename_num(0)
    sample_path = video_dir / f"{video_name}_{sample_num:06d}{image_ext}"
    sample_img = open_image_or_blank(sample_path, None).convert("RGBA")
    tile_w, tile_h = sample_img.size

    font_title, font_header, font_body, font_concept = load_font()

    # 프레임 수 제한
    frame_range = range(T)
    if MAX_FRAMES_PER_VIDEO is not None:
        frame_range = range(min(T, MAX_FRAMES_PER_VIDEO))

    out_dir = out_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 사이드 패널 레이아웃 파라미터 =====
    PANEL_WIDTH = max(int(tile_w * 0.55), 520)     # 패널 폭(이미지 폭 대비)
    GUTTER = 16                                     # 이미지와 패널 사이 간격
    canvas_w = tile_w + GUTTER + PANEL_WIDTH
    canvas_bg_rgba = CANVAS_BG + (255,)

    for t in frame_range:
        base_num = frame_to_filename_num(t)
        img_path = video_dir / f"{video_name}_{base_num:06d}{image_ext}"
        img = open_image_or_blank(img_path, sample_img).convert("RGBA")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.BILINEAR)

        # -------- 데이터 준비 --------
        pred_c = int(preds[t].item())
        gt_c = int(gts[t].item())
        contrib_vec = contribs[t, :, pred_c]  # (64,)

        max_val = float(torch.max(contrib_vec).item())
        thr = 0.9 * max_val
        cand_idx = (contrib_vec >= thr).nonzero(as_tuple=True)[0].tolist()
        if not cand_idx:
            cand_idx = [int(torch.argmax(contrib_vec).item())]
        cand_idx = sorted(cand_idx, key=lambda i: float(contrib_vec[i].item()), reverse=True)

        # -------- 패널 내용 줄바꿈/높이 계산 --------
        tmp = Image.new("RGBA", (10, 10))
        td = ImageDraw.Draw(tmp)
        max_text_w = PANEL_WIDTH - 2 * (PADDING + PANEL_INNER_PAD)

        title_line = f"{video_name}  ·  t={t}  ·  {video_name}_{base_num:06d}{image_ext}"
        title_h = _textsize(td, title_line, font_title)[1]
        info_lines = [
            f"Predicted: {format_phase(pred_c)}    |    GT: {format_phase(gt_c)}",
            # f"Threshold (0.9 × max): {thr:.6f}    ·    max contrib: {max_val:.6f}",
        ]

        info_h = 0
        for line in info_lines:
            for wl in _wrap_text_by_width(td, line, font_concept, max_text_w):
                info_h += _textsize(td, wl, font_concept)[1] + LINE_GAP
        if info_h: info_h -= LINE_GAP

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

            # 줄바꿈된 콘셉트 라인 높이 측정
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
        if neuron_h: neuron_h -= SECTION_GAP

        # 패널 높이 = 이미지 높이(고정). 내용이 더 크면 캔버스 높이를 늘림.
        content_h = int(PANEL_INNER_PAD + title_h + SECTION_GAP +
                     info_h + SECTION_GAP + neuron_h*1.5 + PANEL_INNER_PAD)
        canvas_h = max(tile_h, content_h)  # 사이드 패널을 이미지 높이 이상으로
        canvas = Image.new("RGBA", (canvas_w, canvas_h), canvas_bg_rgba)

        # ---- 왼쪽: 이미지 ----
        # 세로 가운데 정렬
        img_y = (canvas_h - tile_h) // 2
        canvas.alpha_composite(img, (0, img_y))

        # ---- 오른쪽: 패널 ----
        panel_x0 = tile_w + GUTTER
        panel_y0 = 0
        panel_x1 = panel_x0 + PANEL_WIDTH
        panel_y1 = canvas_h
        _draw_rounded_panel(canvas, (panel_x0 + PADDING, panel_y0 + PADDING, panel_x1 - PADDING, panel_y1 - PADDING),
                            PANEL_RADIUS, PANEL_BG)

        draw = ImageDraw.Draw(canvas)
        cur_x = panel_x0 + PADDING + PANEL_INNER_PAD
        cur_y = panel_y0 + PADDING + PANEL_INNER_PAD

        # 타이틀
        draw.text((cur_x, cur_y), title_line, font=font_title, fill=TEXT_PRIMARY)
        cur_y += title_h + SECTION_GAP

        # Info
        for line in info_lines:
            for wl in _wrap_text_by_width(draw, line, font_concept, max_text_w):
                draw.text((cur_x, cur_y), wl, font=font_concept, fill=TEXT_SECONDARY)
                cur_y += _textsize(draw, wl, font_concept)[1] + LINE_GAP
        cur_y += SECTION_GAP

        # Neurons
        for block in measured_neurons:
            draw.text((cur_x, cur_y), block["header"], font=font_header, fill=ACCENT)
            cur_y += _textsize(draw, block["header"], font_header)[1] + LINE_GAP

            for name_for_color, wrapped in zip(block["concept_names_for_color"], block["concept_wrapped"]):
                bcol = _concept_color(name_for_color)
                # 불릿
                draw.ellipse(
                    (cur_x, cur_y + 6, cur_x + 2*BULLET_RADIUS, cur_y + 6 + 2*BULLET_RADIUS),
                    fill=bcol, outline=None
                )
                # 첫 줄
                first = wrapped[0] if wrapped else ""
                draw.text((cur_x + 2*BULLET_RADIUS + 8, cur_y), first, font=font_concept, fill=TEXT_PRIMARY)
                line_h = max(_textsize(draw, first, font_body)[1], 2*BULLET_RADIUS + 6)
                cur_y += line_h + LINE_GAP
                # 이어지는 줄
                for cont in wrapped[1:]:
                    draw.text((cur_x + 2*BULLET_RADIUS + 8, cur_y), cont, font=font_concept, fill=TEXT_PRIMARY)
                    cur_y += _textsize(draw, cont, font_body)[1] + LINE_GAP

            cur_y += SECTION_GAP

        # 저장
        out_path = out_dir / f"{video_name}_{base_num:06d}.png"
        canvas.convert("RGB").save(out_path, format="PNG")


def main():
    contrib_data = load_contributions(CONTRIB_PKL)
    neuron_map = load_neuron_json(NEURON_JSON)

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

        visualize_video(
            video_idx=vid_i,
            contribs=contribs,
            preds=preds,
            gts=gts,
            neuron_map=neuron_map,
            image_root=IMAGE_ROOT,
            out_root=OUT_ROOT,
            image_ext=IMAGE_EXT,
        )

    print("✅ Done. Results saved under:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
