import requests
from bs4 import BeautifulSoup
import streamlit as st

mapping = {
    'Seoul': '서울',
    'Busan': '부산',
    'Daegu': '대구',
    'Incheon': '인천',
    'Gwangju': '광주',
    'Daejeon': '대전',
    'Ulsan': '울산',
    'Sejong': '세종',
    'Yongin': '용인',
    'Suwon': '수원',
    'Changwon': '창원',
    'Jeonju': '전주',
    'Cheongju': '청주',
    'Jeju': '제주',
    'Gangneung': '강릉',
    'Jecheon': '제천',
    'Chuncheon': '춘천',
    'Gyeongju': '경주',
}

# 사용자의 위치 정보 가져오기
def get_location():
    try:
        # IP 주소 확인 (ipinfo.io 무료 API 사용 예제)
        ip_response = requests.get("https://ipinfo.io/json")
        ip_data = ip_response.json()

        # 위치 정보 추출
        city = ip_data.get("city")
        if city:
            city_kor = mapping.get(city, city)  # 한글로 변환된 도시명
            return city_kor
        else:

            return None

    except requests.RequestException:
        
        return None

class NaverWeatherCrawler:
    def __init__(self, location):
        self.base_url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query="
        self.location = location

    def get_weather(self):
        url = self.base_url + self.location + " 날씨"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            weather_info = self.parse_weather(soup)
            return weather_info
        else:
            st.error("날씨 정보를 가져오는 데 실패했습니다.")
            return None

    def parse_weather(self, soup):
        weather = {}
        temp_element = soup.find('div', {'class': 'temperature_text'})
        condition_element = soup.find('span', {'class': 'weather before_slash'})
        location_element = soup.find('h2', {'class': 'title'})

        weather['temperature'] = temp_element.text if temp_element else 'N/A'
        weather['temperature'] = weather['temperature'].replace('현재 온도', '').replace('°', '').strip()
        weather['condition'] = condition_element.text if condition_element else 'N/A'
        weather['location'] = location_element.text if location_element else 'N/A'
        return weather

def get_weather(location):
    crawler = NaverWeatherCrawler(location=location)
    return crawler.get_weather()
