import streamlit as st
from db import initialize_database, get_api_key

# 데이터베이스 테이블 생성
initialize_database()

# 로그인 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""

def logout():
    st.session_state.logged_in = False
    st.session_state['id'] = ""
    st.session_state['username'] = ""
    st.session_state['api_key'] = ""
    st.rerun()

login_page = st.Page("services/login.py", title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
chatbot = st.Page(
    "services/chatbot.py", title="챗봇", icon=":material/chat:", default=True
    )
weather = st.Page(
    "services/weather.py", title="날씨", icon=":material/wb_sunny:"
    )
user_info = st.Page(
    "services/user.py", title="개인 정보", icon=":material/notification_important:"
    )
upload = st.Page(
    "services/upload.py", title="이미지 업로드", icon=":material/image_search:"
    )
getdb = st.Page(
    "services/getdb.py", title="의류 정보 보기", icon=":material/visibility:"
    )

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [chatbot, weather, user_info, upload, getdb],
        }
    )
else:
    pg = st.navigation([login_page])
    
pg.run()