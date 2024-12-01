import streamlit as st
import sqlite3
from datetime import datetime
from db import get_personal_info, add_personal_info, update_personal_info, delete_personal_info  # 삭제 함수 추가

def get_user_info():
    # 세션 상태에서 사용자 ID 가져오기
    user_id = st.session_state.get('id')
    username = st.session_state.get('username')

    if user_id is None:
        st.error("로그인이 필요합니다.")
        return

    st.title(f"{username}님의 개인정보")

    # 데이터베이스에서 기존 개인 정보 가져오기
    existing_info = get_personal_info(user_id)

    # 수정 모드 플래그 설정
    if 'editing' not in st.session_state:
        st.session_state['editing'] = False  # 기본적으로 수정 모드는 False

    # 수정 버튼 클릭 시 수정 모드 활성화 및 페이지 리프레시
    if st.session_state['editing']:
        # 입력 필드 표시 (수정 모드일 때)
        st.write("개인정보를 입력하거나 수정하세요.")
        name = st.text_input("이름", value=existing_info[1] if existing_info else "", key="name_input")
        birthdate = st.date_input(
            "생년월일",
            datetime.strptime(existing_info[2], '%Y-%m-%d') if existing_info else datetime(2000, 1, 1),
            key="birthdate_input"
        )
        gender_options = ["남성", "여성", "기타"]
        gender = st.selectbox(
            "성별", gender_options,
            index=gender_options.index(existing_info[3]) if existing_info else 0,
            key="gender_select"
        )
        height = st.number_input("키 (cm)", min_value=0, value=int(existing_info[4]) if existing_info else 0, key="height_input")
        weight = st.number_input("체중 (kg)", min_value=0, value=int(existing_info[5]) if existing_info else 0, key="weight_input")

        personal_color = st.text_input(
            "퍼스널컬러", value=existing_info[6] if existing_info else "",
            placeholder="모를 시 빈칸으로 두시오",
            key="personal_color_input"
        )
        mbti = st.text_input(
            "MBTI", value=existing_info[7] if existing_info else "",
            placeholder="모를 시 빈칸으로 두시오",
            key="mbti_input"
        )

        col1, col2 = st.columns(2)
        
        # 데이터 저장 버튼
        with col1:
            if st.button("저장"):
                birthdate_str = birthdate.strftime('%Y-%m-%d')

                if existing_info:
                    # 기존 정보가 있으면 업데이트
                    update_personal_info(user_id, name, birthdate_str, gender, height, weight, personal_color, mbti)
                    st.success("데이터가 성공적으로 업데이트되었습니다.")
                else:
                    # 기존 정보가 없으면 새로 추가
                    add_personal_info(user_id, name, birthdate_str, gender, height, weight, personal_color, mbti)
                    st.success("데이터가 성공적으로 저장되었습니다.")

                # 수정 모드 종료
                st.session_state['editing'] = False
                st.rerun()  # 페이지 리프레시

        with col2:
            # 삭제 버튼
            if st.button("정보 삭제"):
                delete_personal_info(user_id)  # 사용자 정보 삭제 함수 호출
                st.success("정보가 삭제되었습니다.")
                st.session_state['editing'] = False
                st.rerun()  # 페이지 리프레시

    else:
        # 수정 모드가 아닐 때 저장된 데이터 보여주기
        if existing_info:
            st.write("저장된 개인정보:")
            st.write(f"**이름:** {existing_info[1]}")
            st.write(f"**생년월일:** {existing_info[2]}")
            st.write(f"**성별:** {existing_info[3]}")
            st.write(f"**키:** {int(existing_info[4])} cm")  # 정수로 변환
            st.write(f"**체중:** {int(existing_info[5])} kg")  # 정수로 변환
            st.write(f"**퍼스널컬러:** {existing_info[6]}")
            st.write(f"**MBTI:** {existing_info[7]}")

            # 수정 버튼
            if st.button("개인정보 수정"):
                st.session_state['editing'] = True
                st.rerun()  # 페이지 리프레시

            
       
        else:
            # 저장된 정보가 없을 때 경고 메시지 표시
            st.warning("저장된 정보가 없습니다. 정보를 입력해주세요.")
            st.session_state['editing'] = True  # 입력 모드 활성화
            st.rerun()  # 페이지 리프레시

get_user_info()
