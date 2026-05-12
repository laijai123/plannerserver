"""
시간표 이미지 디버그 스크립트.
사용법: python debug_ocr.py <이미지 경로>
결과 이미지가 debug_output/ 폴더에 저장됩니다.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from ocr_utils import (
    _load_image, _find_grid_left, _read_time_labels, _fit_time_scale,
    _detect_day_columns, _detect_blocks_in_column,
    DAY_NAMES,
)

if len(sys.argv) < 2:
    print("사용법: python debug_ocr.py <이미지 경로>")
    sys.exit(1)

out_dir = Path("debug_output")
out_dir.mkdir(exist_ok=True)

with open(sys.argv[1], "rb") as f:
    image_bytes = f.read()
image = _load_image(image_bytes=image_bytes)
img_h, img_w = image.shape[:2]
print(f"이미지 크기: {img_w}x{img_h}")
grid_left = _find_grid_left(image)
print(f"그리드 시작 x: {grid_left:.1f}px  ({grid_left/img_w*100:.1f}% of width)")

# ── 1. 시간 눈금 감지 ─────────────────────────────────────────────────────────
marks = _read_time_labels(image)
print(f"\n[시간 눈금] {len(marks)}개 감지:")
for y, h24 in marks:
    print(f"  y={y:.1f}px  →  {h24:02d}:00")

vis_marks = image.copy()
if len(marks) >= 2:
    slope, intercept = _fit_time_scale(marks)
    print(f"  slope={slope:.4f} min/px, intercept={intercept:.1f} min")
    for y, h24 in marks:
        cv2.circle(vis_marks, (int(img_w * 0.06), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(vis_marks, f"{h24}:00", (int(img_w * 0.07), int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
else:
    print("  ⚠️  눈금 2개 미만 → fallback 사용")
    top_y, bot_y = img_h * 0.10, img_h * 0.95
    slope = (22 * 60 - 9 * 60) / max(bot_y - top_y, 1)
    intercept = 9 * 60 - slope * top_y
cv2.imwrite(str(out_dir / "1_time_marks.png"), vis_marks)

# ── 2. 열 감지 ───────────────────────────────────────────────────────────────
n_days = 5
day_centers = _detect_day_columns(image, n_days=n_days)
col_half = (
    (day_centers[-1] - day_centers[0]) / (n_days - 1) / 2
    if len(day_centers) >= 2
    else img_w * 0.44 / n_days
)
print(f"\n[열 감지] col_half={col_half:.1f}px")
for i, cx in enumerate(day_centers):
    print(f"  {DAY_NAMES[i]}: center={cx:.1f}px  x1={cx-col_half:.0f}  x2={cx+col_half:.0f}")

vis_cols = image.copy()
for i, cx in enumerate(day_centers):
    x1 = max(0, int(cx - col_half))
    x2 = min(img_w, int(cx + col_half))
    cv2.line(vis_cols, (x1, 0), (x1, img_h), (255, 0, 0), 2)
    cv2.line(vis_cols, (x2, 0), (x2, img_h), (255, 0, 0), 2)
    cv2.putText(vis_cols, DAY_NAMES[i], (x1 + 4, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.imwrite(str(out_dir / "2_columns.png"), vis_cols)

# ── 3. 블록 감지 ─────────────────────────────────────────────────────────────
CROP_MARGIN = 6
vis_blocks = image.copy()
total_blocks = 0
for i, center in enumerate(day_centers):
    x1 = max(0, int(center - col_half))
    x2 = min(img_w, int(center + col_half))
    col_crop = image[:, x1 + CROP_MARGIN:x2 - CROP_MARGIN]
    blocks = _detect_blocks_in_column(col_crop)
    print(f"\n  {DAY_NAMES[i]}: {len(blocks)}개 블록")
    for b in blocks:
        y, h = b["y"], b["h"]
        start_min = slope * y + intercept
        end_min = slope * (y + h) + intercept
        start_h, start_m = int(start_min) // 60, int(start_min) % 60
        end_h, end_m = int(end_min) // 60, int(end_min) % 60
        print(f"    y={y}~{y+h}px  →  {start_h:02d}:{start_m:02d}~{end_h:02d}:{end_m:02d}")
        abs_x = x1 + b["x"]
        abs_y = b["y"]
        cv2.rectangle(vis_blocks, (abs_x, abs_y), (abs_x + b["w"], abs_y + h),
                      (0, 200, 0), 2)
        cv2.putText(vis_blocks, DAY_NAMES[i], (abs_x + 2, abs_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    total_blocks += len(blocks)

print(f"\n총 {total_blocks}개 블록 감지")
cv2.imwrite(str(out_dir / "3_blocks.png"), vis_blocks)

print(f"\n결과 이미지 저장됨: {out_dir}/")
print("  1_time_marks.png - 시간 눈금 감지 결과")
print("  2_columns.png    - 열 경계 감지 결과")
print("  3_blocks.png     - 블록 감지 결과")
