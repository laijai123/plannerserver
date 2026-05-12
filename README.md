# plannerserver & OCR using claude
SNUPL 앱 용 서버

## OCR timetable endpoint

This server now supports timetable screenshot OCR using local OCR (OpenCV + Tesseract).

Required environment variables:

- None for OCR. If you use `/ocr/timetable` on Render, the Docker image installs Tesseract and the Korean/English language packs.

API:

- `POST /ocr/timetable`
- multipart form field: `image` or `img`
- or JSON body: `{"image_base64": "..."}`
- or JSON body: `{"image_url": "https://..."}`

Response shape:

```json
{
	"parsed": {
		"월": [
			{
				"name": "...",
				"start": "12:30",
				"end": "13:45",
				"professor": "...",
				"location": "..."
			}
		]
	}
}
```
