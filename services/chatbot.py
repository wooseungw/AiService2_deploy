import streamlit as st
from db import update_api_key, get_api_key, get_personal_info, get_user_images
import os
import time

from chatbot_class import GPT
from weather_func import get_weather, get_location

# 스트리밍 데이터를 생성하는 함수
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# 대화 기록 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# API 키 초기화
# API 키 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = get_api_key(st.session_state.get('username', ''))


# 챗봇 객체를 한 번만 생성하도록 변경
if 'chatbot' not in st.session_state:
    user_id = st.session_state.get('id', '')
    existing_info = get_personal_info(user_id)
    weather = get_weather(get_location())

    # 개인정보가 없으면 안내 메시지 출력
    if not existing_info:
        st.warning("개인정보가 없습니다. 개인정보를 입력해주세요.")

    # GPT 챗봇 객체를 세션 상태에 저장
    st.session_state.chatbot = GPT(
        api_key=st.session_state.api_key,
        user_info=existing_info,
        weather=weather,
    )

# 챗봇 페이지를 보여주는 함수
def show_chatbot_page():
    st.title("패션 도우미")

    # 유저 이미지 불러오기
    user_id = st.session_state.get('id', '')
    user_images = get_user_images(user_id)
    items_per_page = 3
    total_pages = (len(user_images) + items_per_page - 1) // items_per_page
    if total_pages > 1:
        page = st.slider('페이지', 1, total_pages, 1)
    else:
        page = 1

    # 해당 페이지의 이미지를 가져오기
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_images = user_images[start_idx:end_idx]

    # 이미지 표시 및 선택 기능
    num_columns = 3
    columns = st.columns(num_columns)

    selected_image = None  # 선택된 이미지 초기화
    for index, img in enumerate(current_images):
        image_id, filename, filepath, upload_date = img  # 수정된 부분
        with columns[index % num_columns]:
            st.image(filepath, use_container_width=True)
            # 선택 버튼
            if st.button(f"선택", key=f"select_{index}"):
                selected_image = filepath
                st.session_state.selected_image = filepath  # 선택한 이미지 세션에 저장

    # 사이드바에 API 키 입력 및 저장 기능 추가
    with st.sidebar:
        api_key_input = st.text_input("API 키 입력", value=st.session_state.api_key, type="password")
        if st.button("API 키 저장"):
            st.session_state.api_key = api_key_input
            if 'username' in st.session_state:
                update_api_key(st.session_state['username'], api_key_input)
                st.success("API 키가 저장되었습니다.")
            else:
                st.error("로그인되어 있지 않습니다.")

    # 이전 대화 기록 표시
    for message in st.session_state['chat_history']:
        role = message["role"]
        if role == "system":
            continue
        elif role == "assistant":
            role = "ai"
        with st.chat_message(role):
            contents = message["content"] if isinstance(message["content"], list) else [message["content"]]
            for content in contents:
                if content["type"] == "text":
                    st.markdown(content["text"])
                elif content["type"] == "image_url":
                    img_name = os.path.basename(content["image_url"]["url"])
                    st.image(content["image_url"]["url"], width=100)

    # 사용자 입력을 받는 입력창
    prompt = st.chat_input("질문을 입력하세요.")
    
    # img_url 초기화
    img_url = None
    
    # 선택한 이미지가 있으면 함께 전송
    if prompt:
        user_content = [{"type": "text", "text": prompt}]
        
        # 이미지가 선택되었다면 추가
        if 'selected_image' in st.session_state and st.session_state.selected_image:
            img_url = st.session_state.selected_image
            user_content.append({"type": "image_url", "image_url": {"url": img_url}})
        
        # 사용자 입력을 대화 기록에 추가
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_content
        })

        # 사용자 입력 표시
        with st.chat_message("user"):
            for content in user_content:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"], width=100)

        # 챗봇 응답 생성
        response = st.session_state.chatbot.generate(text_prompt=prompt, img_prompt=img_url)

        # 챗봇 응답 표시 및 대화 기록에 추가
        with st.chat_message("ai"):
            response_text = "".join(stream_data(response))
            st.write(response_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })

        # 선택한 이미지 상태 초기화
        st.session_state.selected_image = None

show_chatbot_page()
