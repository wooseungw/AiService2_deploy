import streamlit as st

def render():
    st.title("회원가입")
    
    new_username = st.text_input("새로운 아이디")
    new_password = st.text_input("새로운 비밀번호", type="password")
    confirm_password = st.text_input("비밀번호 확인", type="password")
    
    if st.button("회원가입"):
        if new_password == confirm_password:
            st.success(f"회원가입 성공! 아이디: {new_username}")
        else:
            st.error("비밀번호가 일치하지 않습니다.")