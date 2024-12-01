import streamlit as st
from db import  login_user, register_user, get_api_key

if 'signup' not in st.session_state:
    st.session_state['signup'] = False

# 페이지 전환
def show_login_page():
    st.title("로그인")

    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        user = login_user(username, password)
        if user:

            st.session_state['logged_in'] = True
            st.session_state['id'] = user[0]
            st.session_state['username'] = username
            
            # 로그인 성공 시 API 키를 DB에서 불러와 세션에 저장
            try:
                api_key = get_api_key(username)
                if api_key:
                    st.session_state['api_key'] = api_key
            except Exception as e:
                st.error("API 키를 불러오는 중 오류 발생: " + str(e))
            st.success(f"{username}님, 환영합니다!")
            # 메인 페이지로 이동
            st.rerun()  # 페이지 새로고침
        else:
            st.error("로그인 정보가 잘못되었습니다.")

    if st.button("회원가입"):
        st.session_state['signup'] = True
        st.rerun()

def show_signup_page():
    st.title("회원가입")

    username = st.text_input("아이디 (회원가입)")
    password = st.text_input("비밀번호 (회원가입)", type="password")
    confirm_password = st.text_input("비밀번호 확인", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("가입 완료"):
            if not username:
                st.error("아이디를 입력해주세요.")
            elif not password:
                st.error("비밀번호를 입력해주세요.")
            elif password != confirm_password:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                # 회원 등록
                register_user(username, password)
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                
                st.success("회원가입이 완료되었습니다!")
                # 회원가입 완료 후 로그인 페이지로 돌아가기
                st.session_state['signup'] = False
                st.rerun()

    with col2:
        if st.button("취소"):
            st.session_state['signup'] = False
            st.rerun()

# 회원가입 상태인지 로그인 상태인지 체크
if st.session_state.get('signup'):
    show_signup_page()
else:
    show_login_page()