import streamlit as st

# 네비게이션 메뉴에 들어갈 페이지 목록 정의
pages = {
    "Home": "홈 페이지에 오신 것을 환영합니다!",
    "User Guide": "여기는 사용자 가이드 페이지입니다.",
    "API": "API 정보를 제공합니다.",
    "Examples": "예제 페이지입니다."
}

# CSS 스타일링 (상단 고정 네비게이션 바)
st.markdown("""
    <style>
    .top-nav {
        background-color: #0d6efd;
        position: fixed;
        top: 0;
        width: 100%;
        height: 50px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-family: Arial, sans-serif;
        font-size: 18px;
        z-index: 100;
    }
    .nav-link {
        color: white;
        padding: 10px 15px;
        text-decoration: none;
        font-weight: bold;
    }
    .nav-link:hover {
        background-color: #0056b3;
        border-radius: 5px;
    }
    .main-content {
        padding-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# 상단 네비게이션 바 구현
st.markdown('<div class="top-nav">', unsafe_allow_html=True)
for page in pages.keys():
    if st.button(page):
        st.session_state["current_page"] = page
st.markdown('</div>', unsafe_allow_html=True)

# 기본으로 'Home' 페이지 설정
current_page = st.session_state.get("current_page", "Home")

# 네비게이션 바 아래에 내용이 나오도록 상단 여백 추가
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# 현재 선택된 페이지 내용 표시
st.write(pages[current_page])

st.markdown('</div>', unsafe_allow_html=True)
