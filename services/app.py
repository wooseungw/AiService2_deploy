import streamlit as st
from streamlit_option_menu import option_menu
from weather import get_weather
from user import get_user_info

# 상단 네비게이션 메뉴 생성
selected = option_menu(
    menu_title=None,  # 메뉴바 제목 없음
    options=["홈", "날씨 검색", "사용자 정보"],  # 메뉴 옵션
    icons=["house", "cloud-sun", "person"],  # 아이콘 추가
    menu_icon="cast",  # 메뉴바 왼쪽 상단 아이콘
    default_index=0,  # 기본 선택된 메뉴
    orientation="horizontal",  # 메뉴를 상단에 가로로 표시
)

# 각 메뉴에 따른 화면 구성
if selected == "홈":
    st.title("Welcome to the Home Page")
    st.write("This is the home page of the Streamlit app.")
    
elif selected == "날씨 검색":
    try:
        st.write("Fetching weather information...")
        weather_info = get_weather()  # weather.py의 get_weather() 호출
        if weather_info:
            st.write(f"Location: {weather_info['location']}")
            st.write(f"Temperature: {weather_info['temperature']}°C")
            st.write(f"Condition: {weather_info['condition']}")
        else:
            st.warning("No weather information available.")
    except Exception as e:
        st.error(f"Error fetching weather information: {e}")
        
elif selected == "사용자 정보":
    try:
        st.write("Fetching user information...")
        user_info = get_user_info()  # user.py의 get_user_info() 호출
        if user_info:
            st.write(user_info)
        else:
            st.warning("No user information available.")
    except Exception as e:
        st.error(f"Error fetching user information: {e}")
