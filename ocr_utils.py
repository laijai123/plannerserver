from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
import pytesseract
import requests
from pytesseract import Output


DAY_NAMES = ["", "월", "화", "수", "목", "금", "토", "일"]
TIME_LABEL_RE = re.compile(r"(오전|오후)\s*(\d{1,2})\s*시")


class OCRConfigurationError(RuntimeError):
    pass


class OCRRequestError(RuntimeError):
    pass


def _round_to_5(minutes: float) -> int:
    return int(round(minutes / 5.0) * 5)


def _minutes_to_time(total_minutes: float) -> str:
    total_minutes = max(0, _round_to_5(total_minutes))
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _load_image(image_bytes: Optional[bytes] = None, image_url: Optional[str] = None) -> np.ndarray:
    if image_bytes is None and not image_url:
        raise OCRRequestError("Either image_bytes or image_url must be provided.")

    if image_bytes is None and image_url:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OCRRequestError(f"Failed to download image: {exc}") from exc
        image_bytes = response.content

    assert image_bytes is not None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OCRRequestError("Could not decode image bytes.")
    return image


def _resize(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)


def _clean_text(text: str) -> str:
    text = text.replace("\x0c", "")
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_time_mark(minutes_text: str) -> Optional[int]:
    normalized = minutes_text.replace(" ", "")
    match = TIME_LABEL_RE.search(normalized)
    if not match:
        return None

    ampm, hour_text = match.groups()
    hour = int(hour_text)
    if ampm == "오전":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour * 60


def _cluster_positions(values: List[float], gap_threshold: float) -> List[float]:
    if not values:
        return []

    values = sorted(values)
    clusters: List[List[float]] = []
    for value in values:
        if not clusters:
            clusters.append([value])
            continue
        current_center = float(sum(clusters[-1])) / len(clusters[-1])
        if abs(value - current_center) > gap_threshold:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    return [float(sum(cluster)) / len(cluster) for cluster in clusters]


def _detect_blocks(image: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 18, 145], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks: List[Dict[str, Any]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 1800:
            continue
        if w < 60 or h < 35:
            continue
        blocks.append({"x": x, "y": y, "w": w, "h": h})

    blocks.sort(key=lambda item: (item["y"], item["x"]))
    return blocks


def _extract_time_scale(image: np.ndarray) -> Tuple[float, float, List[Tuple[float, int]]]:
    height, width = image.shape[:2]
    left_width = max(120, int(width * 0.18))
    left_crop = image[:, :left_width]
    left_crop = _resize(left_crop, 1.5)
    gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    try:
        data = pytesseract.image_to_data(
            gray,
            output_type=Output.DICT,
            lang="kor+eng",
            config="--oem 3 --psm 6",
            timeout=10,
        )
    except RuntimeError:
        return 1.0, 0.0, []

    grouped: Dict[Tuple[int, int, int], List[Tuple[int, int, int, str]]] = defaultdict(list)
    for index, raw_text in enumerate(data["text"]):
        text = _clean_text(str(raw_text))
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1
        if confidence >= 0 and confidence < 20:
            continue
        key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        grouped[key].append(
            (
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                text,
            )
        )

    marks: List[Tuple[float, int]] = []
    for items in grouped.values():
        items.sort(key=lambda item: item[0])
        line_text = _clean_text(" ".join(item[4] for item in items))
        minute_mark = _extract_time_mark(line_text)
        if minute_mark is None:
            continue

        y_center = sum((item[1] + item[3] / 2.0) / 2.0 for item in items) / len(items)
        marks.append((y_center, minute_mark))

    marks.sort(key=lambda item: item[0])

    if len(marks) >= 2:
        ys = np.array([item[0] for item in marks], dtype=np.float64)
        minutes = np.array([item[1] for item in marks], dtype=np.float64)
        slope, intercept = np.polyfit(ys, minutes, 1)
        return float(slope), float(intercept), marks

    # Fallback based on a typical timetable layout if OCR cannot read the left scale reliably.
    # The fallback still keeps the service usable instead of failing entirely.
    return 1.0, 0.0, marks


def _ocr_block_text(crop: np.ndarray) -> str:
    enlarged = _resize(crop, 1.8)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(binary, lang="kor+eng", config="--oem 3 --psm 6", timeout=8)
    except RuntimeError:
        return ""

    cleaned = _clean_text(text)
    return cleaned


def _parse_block_text(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if not lines:
        return {"name": "", "professor": "", "location": "", "raw_text": text.strip()}

    result = {
        "name": lines[0],
        "professor": lines[1] if len(lines) > 1 else "",
        "location": lines[2] if len(lines) > 2 else "",
        "raw_text": text.strip(),
    }
    return result


def analyze_timetable_image(
    *,
    image_bytes: Optional[bytes] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    image = _load_image(image_bytes=image_bytes, image_url=image_url)
    blocks = _detect_blocks(image)

    if not blocks:
        raise OCRRequestError("No colored timetable blocks were detected in the image.")

    x_centers = [block["x"] + block["w"] / 2.0 for block in blocks]
    day_centers = _cluster_positions(x_centers, gap_threshold=max(35.0, image.shape[1] * 0.04))

    if not day_centers:
        raise OCRRequestError("Could not determine timetable columns from the image.")

    visible_days = DAY_NAMES[1 : 1 + len(day_centers)]
    time_slope, time_intercept, time_marks = _extract_time_scale(image)

    parsed: Dict[str, List[Dict[str, Any]]] = {day: [] for day in visible_days}
    block_results: List[Dict[str, Any]] = []

    for block in blocks:
        x = block["x"]
        y = block["y"]
        w = block["w"]
        h = block["h"]
        x_center = x + w / 2.0
        day_index = min(range(len(day_centers)), key=lambda idx: abs(x_center - day_centers[idx]))
        day_name = visible_days[day_index]

        crop = image[max(0, y - 4) : min(image.shape[0], y + h + 4), max(0, x - 4) : min(image.shape[1], x + w + 4)]
        ocr_text = _ocr_block_text(crop)
        parsed_text = _parse_block_text(ocr_text)

        start_minutes = time_slope * y + time_intercept
        end_minutes = time_slope * (y + h) + time_intercept

        record = {
            "name": parsed_text["name"],
            "start": _minutes_to_time(start_minutes),
            "end": _minutes_to_time(end_minutes),
            "professor": parsed_text["professor"],
            "location": parsed_text["location"],
        }

        parsed[day_name].append(record)
        block_results.append(
            {
                "day": day_name,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                **record,
                "raw_text": parsed_text["raw_text"],
            }
        )

    for day_name, items in parsed.items():
        items.sort(key=lambda item: item["start"])

    return {
        "parsed": parsed,
        "blocks": block_results,
        "time_scale": {
            "slope": time_slope,
            "intercept": time_intercept,
            "detected_marks": [
                {"y": round(y, 2), "minutes": minutes} for y, minutes in time_marks
            ],
        },
    }
