from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
def fetch_everytime_timetable_selenium(url):
	"""
	Selenium을 사용해 동적으로 렌더링된 에브리타임 시간표를 파싱합니다.
	"""
	options = Options()
	options.add_argument('--headless')
	options.add_argument('--no-sandbox')
	options.add_argument('--disable-dev-shm-usage')
	driver = webdriver.Chrome(options=options)
	driver.get(url)
	time.sleep(2)  # JS 렌더링 대기 (필요시 조정)
	html = driver.page_source
	driver.quit()

	soup = BeautifulSoup(html, 'html.parser')
	days = ['월', '화', '수', '목', '금']
	timetable = {day: [] for day in days}

	table = soup.find('div', class_='tablebody')
	cols_list = table.find_all('div', class_='cols') if table else []
	for i, cols in enumerate(cols_list[:5]):
		subjects = cols.find_all('div', class_='subject')
		for subj in subjects:
			name = subj.find('h3').get_text(strip=True) if subj.find('h3') else ''
			prof = subj.find('em').get_text(strip=True) if subj.find('em') else ''
			room = subj.find('span').get_text(strip=True) if subj.find('span') else ''
			timetable[days[i]].append({
				'name': name,
				'professor': prof,
				'room': room
			})
	return timetable
import requests
from bs4 import BeautifulSoup
import re

def fetch_everytime_timetable(url):
	"""
	Selenium을 사용해 동적으로 렌더링된 에브리타임 시간표를 파싱합니다.
	"""
	from selenium import webdriver
	from selenium.webdriver.chrome.options import Options
	from selenium.webdriver.common.by import By
	from selenium.webdriver.support.ui import WebDriverWait
	from selenium.webdriver.support import expected_conditions as EC
	import time
	options = Options()
	options.add_argument('--headless')
	options.add_argument('--no-sandbox')
	options.add_argument('--disable-dev-shm-usage')
	options.add_argument('--window-size=1920,1080')
	options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
	driver = webdriver.Chrome(options=options)
	driver.get(url)

	try:
		WebDriverWait(driver, 10).until(
			EC.presence_of_element_located((By.CLASS_NAME, 'tablebody'))
		)
		html = driver.page_source
	finally:
		driver.quit()

	soup = BeautifulSoup(html, 'html.parser')
	days = ['월', '화', '수', '목', '금', '토', '일']
	timetable = {day: [] for day in days}

	div_tablebody = soup.find('div', class_='tablebody')
	if not div_tablebody:
		raise ValueError('시간표 div(tablebody)를 찾을 수 없습니다.')
	table = div_tablebody.find('table', class_='tablebody')
	if not table:
		raise ValueError('시간표 table(tablebody)를 찾을 수 없습니다.')
	tds = table.find_all('td')

	def round_to_15(minutes):
		# 가장 가까운 15분 단위로 보정
		return int(round(minutes / 15.0) * 15)

	def px_to_minutes(px):
		# 현재 에브리타임 UI에서는 subject top 픽셀이 자정(00:00) 기준 분과 거의 1:1로 매핑됨
		raw_minutes = px
		return round_to_15(raw_minutes)

	def get_minutes_from_height(height):
		raw_duration = height
		duration = round_to_15(raw_duration)
		return max(15, duration)

	def minutes_to_time(total_minutes):
		total_minutes = max(0, total_minutes)
		hour = total_minutes // 60
		minute = total_minutes % 60
		return f"{hour:02d}:{minute:02d}"

	for i, td in enumerate(tds[:7]):
		cols = td.find('div', class_='cols')
		if not cols:
			continue
		subjects = cols.find_all('div', class_='subject')
		for subj in subjects:
			name = subj.find('h3').get_text(strip=True) if subj.find('h3') else ''
			prof = subj.find('em').get_text(strip=True) if subj.find('em') else ''
			room = subj.find('span').get_text(strip=True) if subj.find('span') else ''
			style = subj.get('style', '')
			top_match = re.search(r'top:\s*(\d+)px', style)
			height_match = re.search(r'height:\s*(\d+)px', style)
			if top_match and height_match:
				top_px = int(top_match.group(1))
				height_px = int(height_match.group(1))
				start_minutes = px_to_minutes(top_px)
				duration_min = get_minutes_from_height(height_px)
				end_minutes = start_minutes + duration_min
				start_time = minutes_to_time(start_minutes)
				end_time = minutes_to_time(end_minutes)
				timetable[days[i]].append({
					'name': name,
					'start': start_time,
					'end': end_time
				})
	return timetable

def save_timetable(timetable, filename='timetable.json'):
	"""
	시간표 데이터를 파일로 저장합니다.
	"""
	import json
	with open(filename, 'w', encoding='utf-8') as f:
		json.dump(timetable, f, ensure_ascii=False, indent=2)

def main():
	url = input('에브리타임 시간표 URL을 입력하세요: ')
	try:
		timetable = fetch_everytime_timetable(url)
		save_timetable(timetable)
		print('시간표가 timetable.json 파일에 저장되었습니다.')
	except Exception as e:
		print(f'에러 발생: {e}')

if __name__ == '__main__':
	main()