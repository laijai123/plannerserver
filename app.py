from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re


app = Flask(__name__)


def fetch_everytime_timetable(url: str) -> dict:
    days = ["월", "화", "수", "목", "금", "토", "일"]
    timetable = {day: [] for day in days}

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Render(Docker)에서 설치된 Chromium 경로
    options.binary_location = "/usr/bin/chromium"
    service = Service("/usr/bin/chromedriver")

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tablebody"))
        )
        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    div_tablebody = soup.find("div", class_="tablebody")
    if not div_tablebody:
        raise ValueError("시간표 div(tablebody)를 찾을 수 없습니다.")

    table = div_tablebody.find("table", class_="tablebody")
    if not table:
        raise ValueError("시간표 table(tablebody)를 찾을 수 없습니다.")

    tds = table.find_all("td")

    def round_to_15(minutes: float) -> int:
        return int(round(minutes / 15.0) * 15)

    def px_to_minutes(px: int) -> int:
        # 현재 시간표 UI에서 top px와 분 단위를 1:1로 매핑 후 15분 단위 보정
        return round_to_15(px)

    def duration_from_height(height: int) -> int:
        return max(15, round_to_15(height))

    def minutes_to_time(total_minutes: int) -> str:
        total_minutes = max(0, total_minutes)
        hour = total_minutes // 60
        minute = total_minutes % 60
        return f"{hour:02d}:{minute:02d}"

    for i, td in enumerate(tds[:7]):
        cols = td.find("div", class_="cols")
        if not cols:
            continue

        subjects = cols.find_all("div", class_="subject")
        for subj in subjects:
            name_node = subj.find("h3")
            name = name_node.get_text(strip=True) if name_node else ""

            style = subj.get("style", "")
            top_match = re.search(r"top:\s*(\d+)px", style)
            height_match = re.search(r"height:\s*(\d+)px", style)
            if not (top_match and height_match):
                continue

            top_px = int(top_match.group(1))
            height_px = int(height_match.group(1))

            start_minutes = px_to_minutes(top_px)
            end_minutes = start_minutes + duration_from_height(height_px)

            timetable[days[i]].append(
                {
                    "name": name,
                    "start": minutes_to_time(start_minutes),
                    "end": minutes_to_time(end_minutes),
                }
            )

    return timetable


@app.get("/health")
def health() -> tuple:
    return jsonify({"ok": True}), 200


@app.post("/timetable")
def timetable() -> tuple:
    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "").strip()

    if not url:
        return jsonify({"error": "url is required"}), 400

    if not url.startswith("https://everytime.kr/@"):
        return jsonify({"error": "invalid everytime url format"}), 400

    try:
        data = fetch_everytime_timetable(url)
        return jsonify(data), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
