import traceback

from flask import Flask, jsonify, request

from ocr_utils import (
    OCRConfigurationError,
    OCRRequestError,
    analyze_timetable_hybrid,
    to_timetable_json,
)

app = Flask(__name__)


@app.get("/health")
def health() -> tuple:
    return jsonify({"ok": True}), 200


@app.post("/ocr/timetable")
def ocr_timetable() -> tuple:
    image_bytes = None
    image_url = None

    if "file" in request.files:
        image_bytes = request.files["file"].read()
        if not image_bytes:
            return jsonify({"error": "업로드된 파일이 비어 있습니다."}), 400
    else:
        payload = request.get_json(silent=True) or {}
        image_url = (payload.get("image_url") or payload.get("url") or "").strip() or None
        if not image_url:
            return jsonify({"error": "file 또는 image_url이 필요합니다."}), 400

    try:
        result = analyze_timetable_hybrid(
            image_bytes=image_bytes,
            image_url=image_url,
        )
        return jsonify(to_timetable_json(result)), 200
    except OCRConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500
    except OCRRequestError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
