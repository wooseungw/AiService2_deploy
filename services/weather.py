import streamlit as st
from weather_func import get_location, get_weather

def add_weather_comments(weather):
    temperature = weather['temperature']
    condition = weather['condition']

    try:
        temp_value = float(temperature)
    except ValueError:
        temp_value = None

    if temp_value is not None:
        if temp_value < 0:
            st.warning("날씨가 매우 🥶춥습니다. 외부 활동에 주의하세요!⚠️")
        elif temp_value < 15:
            st.warning("날씨가 🍃쌀쌀합니다. 외출 시 따뜻한 옷을 입으세요!🧥")
        elif temp_value < 25:
            st.warning("날씨가 👍쾌적합니다. 즐거운 하루 되세요!😀")
        elif temp_value >= 30:
            st.warning("날씨가 무척 🥵덥습니다. 수분을 충분히 섭취하세요🧃")

    if '비' in condition:
        st.warning("🌧️비가 오고 있습니다. 우산을 챙기세요!☔")
    elif '눈' in condition:
        st.warning("❄️눈이 오고 있습니다. 미끄럼에 주의하세요!🛝")

def get_weather_streamlit():
    # Streamlit에서 사용자로부터 위치 입력 받기
    st.title("날씨 정보")
    location = st.text_input("위치를 입력하세요", "서울")

    # 위치 정보를 위도와 경도로 변환
    lat, lon = get_location()
    if lat is not None and lon is not None:
        weather_info = get_weather(lat, lon)

        if weather_info:
            st.write(f"위치: {location}")
            st.write(f"온도: {weather_info['temperature']}°C")
            st.write(f"날씨 상태: {weather_info['condition']}")

            # 온도와 날씨 상태에 따른 주의점 추가
            add_weather_comments(weather_info)
        else:
            st.error("날씨 정보를 가져오지 못했습니다.")
    else:
        st.error("위치 정보를 가져오지 못했습니다.")

get_weather_streamlit()