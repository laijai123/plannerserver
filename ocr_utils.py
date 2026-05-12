from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from pytesseract import Output


DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


class OCRConfigurationError(RuntimeError):
    pass


class OCRRequestError(RuntimeError):
    pass


# ── image loading ─────────────────────────────────────────────────────────────


def _load_image(
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
) -> np.ndarray:
    if image_bytes is None and image_url:
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            image_bytes = r.content
        except requests.RequestException as exc:
            raise OCRRequestError(f"이미지 다운로드 실패: {exc}") from exc
    if image_bytes is None:
        raise OCRRequestError("image_bytes 또는 image_url이 필요합니다.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise OCRRequestError("이미지 디코드 실패.")
    return img


# ── time scale detection ──────────────────────────────────────────────────────


def _to_binary_inv(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Dark-bg images (dark mode): BINARY_INV turns white text → black on white bg for Tesseract.
    # Light-bg images (light mode): BINARY keeps dark text → black on white bg for Tesseract.
    thresh_type = (
        cv2.THRESH_BINARY_INV if float(np.mean(gray)) < 128 else cv2.THRESH_BINARY
    ) + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, 0, 255, thresh_type)
    return binary


def _find_grid_left(image: np.ndarray) -> float:
    """
    Return the x-coordinate where the day-column grid starts.
    The time-label strip has near-zero chroma (white digits on black bg);
    scan left-to-right and return the first x where ANY colored content appears.
    """
    img_h, img_w = image.shape[:2]
    bgr = image.astype(np.int16)
    chroma = (np.max(bgr, axis=2) - np.min(bgr, axis=2)).astype(np.uint8)
    body = chroma[int(img_h * 0.08):, :]
    col_density = np.sum(body > 15, axis=0).astype(float)
    # Tiny kernel — keep spatial precision so we don't smear the transition
    k = max(3, img_w // 200)
    col_density = np.convolve(col_density, np.ones(k) / k, mode="same")
    # Use 3 % of content-area mean as threshold (was 10 %, which overshoots
    # when the first column has sparse blocks and later columns are denser)
    content_mean = col_density[int(img_w * 0.30): int(img_w * 0.90)].mean()
    threshold = max(2.0, content_mean * 0.03)
    for x in range(int(img_w * 0.01), int(img_w * 0.25)):
        if col_density[x] > threshold:
            return float(x)
    return img_w * 0.05


def _read_time_labels(image: np.ndarray) -> List[Tuple[float, int]]:
    """Read 12-hour labels from the left strip; return [(y_px, hour_24)]."""
    img_h, img_w = image.shape[:2]
    # Use detected grid boundary so the strip doesn't spill into course blocks
    grid_left = _find_grid_left(image)
    strip_w = max(40, int(grid_left * 1.5))
    strip = image[:, :strip_w]
    binary = _to_binary_inv(strip, scale=3.0)

    try:
        data = pytesseract.image_to_data(
            binary, output_type=Output.DICT, lang="eng",
            config="--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789",
            timeout=10,
        )
    except RuntimeError:
        return []

    raw: List[Tuple[float, int]] = []
    for i, txt in enumerate(data.get("text", [])):
        txt = str(txt).strip()
        if not re.fullmatch(r"\d{1,2}", txt):
            continue
        v = int(txt)
        if not (1 <= v <= 12):
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if 0 <= conf < 10:
            continue
        top = int(data["top"][i])
        # Everytime draws the hour label with its top edge at the gridline.
        # Use the text top (not center) so the fitted intercept is correct.
        raw.append((top / 3.0, v))

    if not raw:
        return []

    raw.sort()
    deduped: List[Tuple[float, int]] = []
    for y, v in raw:
        if deduped and abs(y - deduped[-1][0]) < img_h * 0.04:
            continue
        deduped.append((y, v))

    result: List[Tuple[float, int]] = []
    prev = -1
    for y, v in deduped:
        h24 = v
        while h24 <= prev and h24 + 12 <= 24:
            h24 += 12
        if 6 <= h24 <= 24 and h24 > prev:
            result.append((y, h24))
            prev = h24

    return result


def _fit_time_scale(marks: List[Tuple[float, int]]) -> Tuple[float, float]:
    """Fit t(y) = slope·y + intercept where t is in minutes."""
    if len(marks) < 2:
        return 1.0, 0.0
    ys = np.array([m[0] for m in marks], dtype=np.float64)
    mins = np.array([m[1] * 60.0 for m in marks], dtype=np.float64)
    slope, intercept = np.polyfit(ys, mins, 1)
    # Everytime draws hour labels with their top edge ~10 px below the actual
    # gridline. We use text-top as the mark y, so shift the intercept to
    # compensate: gridline_y = text_top_y - 10  →  intercept += slope * 10.
    intercept += float(slope) * 10.0
    return float(slope), float(intercept)


# ── image annotation ──────────────────────────────────────────────────────────


def _annotate_with_time_grid(image: np.ndarray) -> np.ndarray:
    """
    Draw 24h time labels every 30 min onto the left strip so Claude can read
    exact times instead of guessing from 12h marks.
    """
    img_h, img_w = image.shape[:2]

    marks = _read_time_labels(image)
    if len(marks) >= 2:
        slope, intercept = _fit_time_scale(marks)
    else:
        # Fallback: assume 9:00 at 10% height, 22:00 at 95% height
        top_y, bot_y = img_h * 0.10, img_h * 0.95
        slope = (22 * 60 - 9 * 60) / max(bot_y - top_y, 1)
        intercept = 9 * 60 - slope * top_y

    annotated = image.copy()
    font_scale = max(0.28, min(0.45, img_h / 1800))
    strip_x = int(img_w * 0.13)

    for total_min in range(8 * 60, 23 * 60, 5):
        y_f = (total_min - intercept) / max(abs(slope), 0.001)
        if not (0 <= y_f < img_h):
            continue
        y = int(y_f)
        h, m = total_min // 60, total_min % 60
        if m == 0:
            # Hour mark: full-width blue line + label
            color = (30, 30, 220)
            cv2.line(annotated, (0, y), (strip_x, y), color, 2)
            ty = max(int(font_scale * 30) + 1, y - 2)
            cv2.putText(annotated, f"{h:02d}:{m:02d}", (2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        elif m % 30 == 0:
            # Half-hour mark: medium purple line + label
            color = (160, 30, 200)
            cv2.line(annotated, (0, y), (strip_x, y), color, 1)
            ty = max(int(font_scale * 30) + 1, y - 2)
            cv2.putText(annotated, f"{h:02d}:{m:02d}", (2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, color, 1, cv2.LINE_AA)
        else:
            # 5-minute mark: short line + small label so Claude can read exact time
            color = (100, 100, 100)
            cv2.line(annotated, (0, y), (int(strip_x * 0.5), y), color, 1)
            ty = max(int(font_scale * 20) + 1, y - 1)
            cv2.putText(annotated, f"{h:02d}:{m:02d}", (2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65, color, 1, cv2.LINE_AA)

    return annotated


# ── column / block detection ─────────────────────────────────────────────────


def _detect_day_columns(image: np.ndarray, n_days: int = 5) -> List[float]:
    """Return x-centers of the n_days timetable columns."""
    img_h, img_w = image.shape[:2]

    # 1. Try Hough vertical line detection first (works for light-theme images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(img_h * 0.30), maxLineGap=15)
    vx: List[float] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) <= 4 and abs(y2 - y1) >= img_h * 0.28:
                vx.append((x1 + x2) / 2.0)
    vx.sort()
    gap = img_w * 0.05
    clusters: List[List[float]] = []
    for x in vx:
        if not clusters or x - sum(clusters[-1]) / len(clusters[-1]) > gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    vx_clean = sorted(sum(c) / len(c) for c in clusters)
    if len(vx_clean) >= n_days + 1:
        seps = vx_clean[:n_days + 1]
        spacings = [seps[i + 1] - seps[i] for i in range(len(seps) - 1)]
        mean_sp = sum(spacings) / len(spacings)
        # Accept only if all gaps are within 30% of the mean (evenly spaced grid)
        if all(abs(sp - mean_sp) / max(mean_sp, 1) < 0.30 for sp in spacings):
            return [(seps[i] + seps[i + 1]) / 2.0 for i in range(n_days)]

    # 2. Fallback: locate the time strip width via color density, then divide equally
    left = _find_grid_left(image)
    col_w = (img_w - left) / n_days
    return [left + (i + 0.5) * col_w for i in range(n_days)]


def _detect_blocks_in_column(col_crop: np.ndarray) -> List[Dict[str, int]]:
    """Detect colored course blocks within a single day-column crop.

    Uses a per-row density profile rather than morphological close so that
    (a) a 1-2 px bleed from an adjacent column (< 1 % row density) is ignored,
    and (b) two courses that share a single dark gap row are split correctly.
    """
    img_h, img_w = col_crop.shape[:2]

    bgr = col_crop.astype(np.int16)
    chroma = (np.max(bgr, axis=2) - np.min(bgr, axis=2)).astype(np.uint8)
    hsv = cv2.cvtColor(col_crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]

    # Colored block pixels only.  A white-background (light mode) would give
    # chroma ≈ 0 everywhere, so it is correctly excluded; dark backgrounds
    # (dark mode) are excluded by val < 30.  The row-density threshold keeps
    # any stray border pixels from registering as a real block.
    content = (chroma >= 8) & (val >= 30)
    content[:int(img_h * 0.07), :] = False  # skip column header

    # Row density: fraction [0, 1] of content pixels in each row
    row_density = content.sum(axis=1).astype(float) / max(img_w, 1)

    # At least 5 % of the row must be colored to count as part of a block.
    # A 1-2 px bleed strip from an adjacent column gives < 1 % → ignored.
    THRESH = 0.05
    in_block = row_density > THRESH

    blocks: List[Dict[str, int]] = []
    start: Optional[int] = None
    for y in range(img_h):
        if in_block[y] and start is None:
            start = y
        elif not in_block[y] and start is not None:
            h = y - start
            if 18 <= h <= int(img_h * 0.65):
                blocks.append({"x": 0, "y": start, "w": img_w, "h": h})
            start = None
    if start is not None:
        h = img_h - start
        if 18 <= h <= int(img_h * 0.65):
            blocks.append({"x": 0, "y": start, "w": img_w, "h": h})

    return sorted(blocks, key=lambda b: b["y"])


# ── time snapping ─────────────────────────────────────────────────────────────


def _minutes_to_time(total_minutes: float) -> str:
    snapped = int(round(total_minutes / 5.0) * 5)
    snapped = max(0, min(24 * 60, snapped))
    return f"{snapped // 60:02d}:{snapped % 60:02d}"


def _snap_time(time_str: str) -> str:
    """Round HH:MM string to the nearest 5-minute increment."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", time_str.strip())
    if not m:
        return time_str
    total = int(m.group(1)) * 60 + int(m.group(2))
    snapped = max(0, min(24 * 60, int(round(total / 5.0) * 5)))
    return f"{snapped // 60:02d}:{snapped % 60:02d}"


# ── public API ────────────────────────────────────────────────────────────────


def analyze_timetable_with_vision(
    *,
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> Dict[str, Any]:
    """
    Extract timetable data from a screenshot using Claude Vision.

    Before sending to Claude, the image is annotated with explicit 24h time
    labels (blue = hour, purple = half-hour) derived from the detected time
    scale, giving Claude precise reference points for 5-minute accuracy.
    """
    import anthropic

    if image_bytes is None and image_url:
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            image_bytes = r.content
        except requests.RequestException as exc:
            raise OCRRequestError(f"이미지 다운로드 실패: {exc}") from exc
    if image_bytes is None:
        raise OCRRequestError("image_bytes 또는 image_url이 필요합니다.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise OCRConfigurationError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    # Annotate image with 24h time grid then encode as PNG for Claude
    image_np = _load_image(image_bytes=image_bytes)
    annotated_np = _annotate_with_time_grid(image_np)
    ok, encoded = cv2.imencode(".png", annotated_np)
    send_bytes = encoded.tobytes() if ok else image_bytes

    image_b64 = base64.b64encode(send_bytes).decode()

    prompt = (
        "이 이미지는 에버타임 대학교 시간표입니다.\n"
        "이미지 왼쪽 가장자리에 파란색(정시) 또는 보라색(30분) 숫자로 정확한 시각(HH:MM 24시간제)이 표시되어 있습니다.\n"
        "이 시각 표시를 기준으로 각 강의 블록의 시작·종료 시간을 5분 단위로 정확하게 추출하세요.\n\n"
        "반드시 아래 JSON 형식으로만 답하세요 (설명 없이 JSON만):\n"
        '{"월":[{"name":"강의명","start":"HH:MM","end":"HH:MM"}],'
        '"화":[],"수":[],"목":[],"금":[]}\n\n'
        "규칙: 시간은 24시간제, 강의 없는 요일은 빈 배열, 강의명은 정확히."
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = message.content[0].text.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise OCRRequestError(f"Vision API 응답에서 JSON을 찾을 수 없습니다: {raw[:200]}")
    parsed = json.loads(match.group())

    # Post-process: snap all times to nearest 5-minute boundary
    for day_entries in parsed.values():
        if isinstance(day_entries, list):
            for entry in day_entries:
                if "start" in entry:
                    entry["start"] = _snap_time(entry["start"])
                if "end" in entry:
                    entry["end"] = _snap_time(entry["end"])

    return {
        "parser_version": "vision-v2",
        "engine": f"claude-vision:{model}",
        "parsed": parsed,
    }


def analyze_timetable_hybrid(
    *,
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> Dict[str, Any]:
    """
    Local CV pipeline for precise times + Claude Vision for course names.

    Times come from pixel-level block detection mapped through the Tesseract-
    calibrated time scale, then snapped to 5-minute boundaries.
    Names come from Claude Vision reading each block crop individually.
    """
    import anthropic

    if image_bytes is None and image_url:
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            image_bytes = r.content
        except requests.RequestException as exc:
            raise OCRRequestError(f"이미지 다운로드 실패: {exc}") from exc
    if image_bytes is None:
        raise OCRRequestError("image_bytes 또는 image_url이 필요합니다.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise OCRConfigurationError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    image = _load_image(image_bytes=image_bytes)
    img_h, img_w = image.shape[:2]

    n_days = 5
    day_centers = _detect_day_columns(image, n_days=n_days)
    day_labels = DAY_NAMES[:n_days]

    col_half = (
        (day_centers[-1] - day_centers[0]) / (n_days - 1) / 2
        if len(day_centers) >= 2
        else img_w * 0.44 / n_days
    )

    time_marks = _read_time_labels(image)
    if len(time_marks) >= 2:
        slope, intercept = _fit_time_scale(time_marks)
    else:
        top_y, bot_y = img_h * 0.10, img_h * 0.95
        slope = (22 * 60 - 9 * 60) / max(bot_y - top_y, 1)
        intercept = 9 * 60 - slope * top_y

    # Detect all blocks; compute times locally, collect PNG crops for Claude
    ordered_blocks: List[Dict[str, Any]] = []

    # Trim a few pixels from each column edge so that 1-2 px color bleed from
    # an adjacent block's border does not enter this column's crop.
    CROP_MARGIN = 6
    for i, center in enumerate(day_centers):
        x1 = max(0, int(center - col_half) + CROP_MARGIN)
        x2 = min(img_w, int(center + col_half) - CROP_MARGIN)
        col_crop = image[:, x1:x2]
        col_w = x2 - x1

        for block in _detect_blocks_in_column(col_crop):
            y, h = block["y"], block["h"]
            x_b, w_b = block["x"], block["w"]

            start_min = slope * y + intercept
            end_min = slope * (y + h) + intercept

            cx1 = max(0, x_b - 20)
            cx2 = min(col_w, x_b + w_b + 4)
            crop = col_crop[max(0, y):min(img_h, y + h), cx1:cx2]

            # Upscale small crops so Claude can read the text
            crop_h, crop_w = crop.shape[:2]
            if crop_h > 0 and crop_h < 60:
                scale = max(2.0, 60.0 / crop_h)
                crop = cv2.resize(
                    crop,
                    (int(crop_w * scale), int(crop_h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                continue

            ordered_blocks.append({
                "day": day_labels[i],
                "start": _minutes_to_time(start_min),
                "end": _minutes_to_time(end_min),
                "crop_b64": base64.b64encode(encoded.tobytes()).decode(),
            })

    if not ordered_blocks:
        return {
            "parser_version": "hybrid-v3",
            "engine": f"hybrid:local+claude-vision:{model}",
            "parsed": {d: [] for d in day_labels},
        }

    # Send all crops to Claude in a single call — names only
    image_content: List[Dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b["crop_b64"]},
        }
        for b in ordered_blocks
    ]
    n = len(ordered_blocks)

    def _build_prompt(extra: str = "") -> str:
        return (
            f"위 {n}개의 이미지는 에버타임 시간표 강의 블록입니다. "
            f"이미지 순서 그대로 강의명을 읽어주세요.\n"
            f"반드시 길이가 정확히 {n}인 JSON 배열 하나만 출력하세요 (설명·주석 없이).\n"
            "글자를 읽기 어려운 항목은 \"미인식\"으로 채우세요. 절대 항목을 건너뛰지 마세요.\n"
            f'형식: ["강의명1","강의명2",...]{extra}'
        )

    def _parse_names(text: str) -> List[str]:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        return [str(s).strip() or "미인식" for s in result]

    client = anthropic.Anthropic(api_key=api_key)

    def _call(prompt: str) -> List[str]:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}],
        )
        return _parse_names(msg.content[0].text.strip())

    names = _call(_build_prompt())

    if len(names) != n:
        # Retry once with an explicit length warning
        names = _call(_build_prompt(
            f"\n\n⚠️ 반드시 {n}개: 이전 응답 길이가 맞지 않았습니다. 항목을 절대 건너뛰지 마세요."
        ))

    # Final safeguard: pad or trim so names always aligns with ordered_blocks
    while len(names) < n:
        names.append("미인식")
    names = names[:n]

    merged: Dict[str, List[Dict[str, Any]]] = {d: [] for d in day_labels}
    for i, block in enumerate(ordered_blocks):
        merged[block["day"]].append({
            "name": names[i],
            "start": block["start"],
            "end": block["end"],
        })

    for day in day_labels:
        merged[day].sort(key=lambda e: e["start"])

    return {
        "parser_version": "hybrid-v3",
        "engine": f"hybrid:local+claude-vision:{model}",
        "parsed": merged,
    }


def to_timetable_json(result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return dict(result.get("parsed", {}))
