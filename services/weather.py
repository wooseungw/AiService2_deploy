import streamlit as st
from weather_func import get_location, get_weather

def add_weather_comments(weather):
    temperature = weather.get('temperature', 'N/A')
    condition = weather.get('condition', 'N/A')

    try:
        temp_value = float(temperature.replace('°', '').replace('C', '').strip())
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
            st.warning("날씨가 무척 🥵덥습니다. 수분을 충분히 섭취하세요!🧃")

    if '비' in condition:
        st.warning("🌧️ 비가 오고 있습니다. 우산을 챙기세요!☔")
    elif '눈' in condition:
        st.warning("❄️ 눈이 오고 있습니다. 미끄럼에 주의하세요!🛝")

def get_weather_streamlit():
    st.title("날씨 정보")
    location = st.text_input("위치를 입력하세요", "서울").strip()
    if not location:
        location = "서울"

    weather_info = get_weather(location)

    if weather_info:
        temperature = weather_info.get('temperature', 'N/A')
        condition = weather_info.get('condition', 'N/A')
        location_name = weather_info.get('location', '알 수 없음')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("위치")
            st.write(location_name)
        with col2:
            st.subheader("온도")
            st.write(f"{temperature}°C")

        st.subheader("날씨 상태")
        st.write(condition)

        add_weather_comments({'temperature': temperature, 'condition': condition})
    else:
        st.error("날씨 정보를 불러올 수 없습니다. 네트워크 연결을 확인하거나, 올바른 위치를 입력했는지 확인하세요.")

get_weather_streamlit()
