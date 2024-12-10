import requests
import json
import streamlit as st

# secret.key 파일에서 키를 로드하는 함수
def load_weather_key():
    # secret.key 파일에서 키 로드
    key = st.secrets["weather"]
    return key

API_KEY = load_weather_key()

# 한글 도시명 매핑
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
        # IP 주소를 기반으로 위치 정보 확인 (ipinfo.io 무료 API 사용 예제)
        ip_response = requests.get("https://ipinfo.io/json")
        ip_data = ip_response.json()

        # 위도와 경도 추출
        loc = ip_data.get("loc")
        if loc:
            lat, lon = map(float, loc.split(","))
            return lat, lon
        else:
            print("위치 정보를 가져올 수 없습니다.")
            return 37.5660, 126.9784
    except requests.RequestException as e:
        print(f"위치 정보를 가져오는 데 실패했습니다: {e}")
        return None, None

# 날씨 정보 가져오기
def get_weather(lat, lon):
    try:
        # OpenWeatherMap API를 사용하여 날씨 정보 가져오기
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            if current:
                weather_info = {
                    'temperature': current.get('temp'),
                    'condition': current['weather'][0]['description'] if 'weather' in current else None,
                    'humidity': current.get('humidity'),
                    'wind_speed': current.get('wind_speed')
                }
                return weather_info
            else:
                print("날씨 정보를 가져올 수 없습니다.")
                return None
        else:
            print(f"날씨 정보를 가져오는 데 실패했습니다. 상태 코드: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"날씨 정보를 가져오는 데 실패했습니다: {e}")
        return None

# 통합 테스트
def test_weather():
    lat, lon = get_location()
    if lat is not None and lon is not None:
        weather_info = get_weather(lat, lon)
        if weather_info:
            print(f"현재 위치의 온도: {weather_info['temperature']}°C")
            print(f"현재 위치의 날씨 상태: {weather_info['condition']}")
            print(f"습도: {weather_info['humidity']}%")
            print(f"풍속: {weather_info['wind_speed']}m/s")
        else:
            print("날씨 정보를 가져오지 못했습니다.")
    else:
        print("위치 정보를 가져오지 못했습니다.")

if __name__ == "__main__":
    test_weather()