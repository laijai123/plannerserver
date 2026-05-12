import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from ocr_utils import (
    OCRRequestError,
    analyze_timetable_hybrid,
    analyze_timetable_with_vision,
    to_timetable_json,
)


class TimetableOCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Timetable Vision Tester")
        self.root.geometry("700x540")
        self._image_path = ""
        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        file_row = ttk.Frame(frame)
        file_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(file_row, text="이미지:").pack(side=tk.LEFT)
        self._path_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self._path_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6)
        )
        ttk.Button(file_row, text="Browse", command=self._browse).pack(side=tk.LEFT)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="Hybrid (권장)", command=self._run_hybrid).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Vision API", command=self._run_vision).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Copy Result", command=self._copy).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=(8, 0))

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self._status_var, foreground="#555").pack(anchor=tk.W, pady=(0, 4))

        self._output = tk.Text(frame, wrap=tk.WORD, font=("Courier", 11))
        scrollbar = ttk.Scrollbar(frame, command=self._output.yview)
        self._output.configure(yscrollcommand=scrollbar.set)
        self._output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="시간표 이미지 선택",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")],
        )
        if path:
            self._image_path = path
            self._path_var.set(path)

    def _run_hybrid(self) -> None:
        if not self._image_path or not os.path.isfile(self._image_path):
            self._status_var.set("이미지 파일을 먼저 선택해주세요.")
            return
        self._status_var.set("Hybrid 분석 중...")
        self._output.delete("1.0", tk.END)
        threading.Thread(target=self._worker, args=("hybrid",), daemon=True).start()

    def _run_vision(self) -> None:
        if not self._image_path or not os.path.isfile(self._image_path):
            self._status_var.set("이미지 파일을 먼저 선택해주세요.")
            return
        self._status_var.set("Vision API 분석 중...")
        self._output.delete("1.0", tk.END)
        threading.Thread(target=self._worker, args=("vision",), daemon=True).start()

    def _worker(self, mode: str) -> None:
        try:
            with open(self._image_path, "rb") as f:
                image_bytes = f.read()
            if mode == "hybrid":
                result = analyze_timetable_hybrid(image_bytes=image_bytes)
            else:
                result = analyze_timetable_with_vision(image_bytes=image_bytes)
            timetable = to_timetable_json(result)
            engine = result.get("engine", "")
            formatted = json.dumps(timetable, ensure_ascii=False, indent=2)
            self.root.after(0, self._show_result, f"// engine: {engine}\n{formatted}")
        except OCRRequestError as exc:
            self.root.after(0, self._show_error, str(exc))
        except Exception as exc:
            import traceback
            self.root.after(0, self._show_error, f"{exc}\n\n{traceback.format_exc()}")

    def _show_result(self, text: str) -> None:
        self._status_var.set("완료")
        self._output.delete("1.0", tk.END)
        self._output.insert(tk.END, text)

    def _show_error(self, msg: str) -> None:
        self._status_var.set("오류")
        self._output.delete("1.0", tk.END)
        self._output.insert(tk.END, f"Error: {msg}")

    def _copy(self) -> None:
        text = self._output.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            self._status_var.set("클립보드에 복사됨")

    def _clear(self) -> None:
        self._output.delete("1.0", tk.END)
        self._status_var.set("Ready")


def main() -> None:
    root = tk.Tk()
    TimetableOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
