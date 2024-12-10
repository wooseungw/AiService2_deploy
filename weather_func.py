import requests
import streamlit as st

# secret.key 파일에서 키를 로드하는 함수
def load_weather_key():
    # secret.key 파일에서 키 로드
    key = st.secrets["weather"]
    return key

API_KEY = load_weather_key()
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
            st.session_state['location'] = city_kor
            return city_kor
        else:
            st.error("도시 정보를 가져올 수 없습니다.")
            return None

    except requests.RequestException:
        st.error("위치 정보를 가져오는 데 실패했습니다.")
        return None

def get_weather(location):
    try:
        # OpenWeatherMap API를 사용하여 날씨 정보 가져오기
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric&lang=kr"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'location': data['name'],
                'temperature': data['main']['temp'],
                'condition': data['weather'][0]['description']
            }
            return weather_info
        else:
            st.error("날씨 정보를 가져오는 데 실패했습니다.")
            return None
    except requests.RequestException:
        st.error("날씨 정보를 가져오는 데 실패했습니다.")
        return None