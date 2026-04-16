import base64
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import requests


DEFAULT_ENDPOINT = "https://everytime-timetable-api.onrender.com/ocr/timetable"


class OCRTesterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Timetable OCR Tester")
        self.root.geometry("860x620")

        self.endpoint_var = tk.StringVar(value=DEFAULT_ENDPOINT)
        self.image_path_var = tk.StringVar(value="")
        self.upload_mode_var = tk.StringVar(value="multipart")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="OCR Endpoint").pack(anchor=tk.W)
        ttk.Entry(container, textvariable=self.endpoint_var).pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(container)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Image File").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.image_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        ttk.Button(row, text="Browse", command=self._browse_image).pack(side=tk.LEFT)

        mode_row = ttk.Frame(container)
        mode_row.pack(fill=tk.X, pady=(10, 8))
        ttk.Label(mode_row, text="Send Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="multipart/form-data", value="multipart", variable=self.upload_mode_var).pack(side=tk.LEFT, padx=(8, 6))
        ttk.Radiobutton(mode_row, text="JSON base64", value="base64", variable=self.upload_mode_var).pack(side=tk.LEFT)

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_row, text="Send Request", command=self._send_request).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Clear Output", command=self._clear_output).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(container, textvariable=self.status_var, foreground="#555").pack(anchor=tk.W, pady=(0, 6))

        self.output = tk.Text(container, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select timetable screenshot",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp"), ("All Files", "*.*")],
        )
        if path:
            self.image_path_var.set(path)

    def _clear_output(self) -> None:
        self.output.delete("1.0", tk.END)
        self.status_var.set("Ready")

    def _send_request(self) -> None:
        endpoint = self.endpoint_var.get().strip()
        image_path = self.image_path_var.get().strip()

        if not endpoint:
            messagebox.showwarning("Missing endpoint", "Please enter an OCR endpoint URL.")
            return
        if not image_path:
            messagebox.showwarning("Missing image", "Please choose an image file.")
            return
        if not os.path.isfile(image_path):
            messagebox.showerror("File not found", f"Image file not found:\n{image_path}")
            return

        self.status_var.set("Sending request...")
        self.output.delete("1.0", tk.END)

        thread = threading.Thread(target=self._request_worker, args=(endpoint, image_path), daemon=True)
        thread.start()

    def _request_worker(self, endpoint: str, image_path: str) -> None:
        try:
            mode = self.upload_mode_var.get()
            if mode == "multipart":
                with open(image_path, "rb") as image_file:
                    files = {"image": (os.path.basename(image_path), image_file, "application/octet-stream")}
                    response = requests.post(endpoint, files=files, timeout=420)
            else:
                with open(image_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                payload = {"image_base64": image_base64}
                headers = {"Content-Type": "application/json"}
                response = requests.post(endpoint, data=json.dumps(payload), headers=headers, timeout=420)

            self._show_response(response)
        except Exception as exc:  # pylint: disable=broad-except
            self.root.after(0, self._render_error, str(exc))

    def _show_response(self, response: requests.Response) -> None:
        try:
            parsed = response.json()
            formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            formatted = response.text

        status_line = f"HTTP {response.status_code}"
        self.root.after(0, self._render_output, status_line, formatted)

    def _render_output(self, status: str, body: str) -> None:
        self.status_var.set(status)
        self.output.insert(tk.END, body)

    def _render_error(self, message: str) -> None:
        self.status_var.set("Request failed")
        self.output.insert(tk.END, f"Error: {message}")


def main() -> None:
    root = tk.Tk()
    OCRTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
