import streamlit as st
from db import get_user_images, get_image_attributes
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image

def draw_bbox_on_image(image_path, bboxes, categories):
    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 카테고리별 색상 정의
    category_colors = {
        "outer": (158, 158, 158),  # 회색
        "top": (249, 202, 144),    # 파랑
        "bottom": (167, 214, 165), # 초록
        "onepiece": (200, 104, 186) # 보라
    }
    
    # 각 바운딩 박스 그리기
    for bbox, category in zip(bboxes, categories):
        bbox = [float(coord) for coord in bbox]
        x1, y1, x2, y2 = map(int, bbox)
        
        color = category_colors.get(category.lower(), (158, 158, 158))
        
        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 카테고리 텍스트 추가
        cv2.putText(img, category, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(img)

def render():
    st.title("저장된 의류 속성 정보 보기")

    user_id = st.session_state.get('id')
    if not user_id:
        st.error("로그인이 필요합니다.")
        st.stop()
    else:
        user_id = int(user_id)

    user_images = get_user_images(user_id)
    if not user_images:
        st.write("저장된 이미지가 없습니다.")
        return

    for image in user_images:
            image_id, filename, filepath, upload_date = image
            
            with st.expander(f"📷 {filename} ({upload_date})", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                attributes = get_image_attributes(image_id)
                if attributes:
                    # 바운딩 박스와 카테고리 정보 수집
                    bboxes = []
                    categories = []
                    for attr in attributes:
                        category, bounding_box, confidence, _ = attr
                        bboxes.append(json.loads(bounding_box))
                        categories.append(category)
                    
                    # 바운딩 박스가 그려진 이미지 생성
                    img_with_bbox = draw_bbox_on_image(filepath, bboxes, categories)
                    
                    with col1:
                        st.image(img_with_bbox, width=300)
                else:
                    with col1:
                        st.image(filepath, width=300)
            
            with col2:
                attributes = get_image_attributes(image_id)
                if not attributes:
                    st.write("저장된 속성 정보가 없습니다.")
                    continue

                for idx, attr in enumerate(attributes, 1):
                    category, bounding_box, confidence, attr_json = attr
                    
                    # 카테고리별 부드러운 색상
                    category_colors = {
                        "outer": "🧥 rgba(158, 158, 158, 0.1)",  # 중성적인 회색
                        "top": "👕 rgba(144, 202, 249, 0.1)",    # 부드러운 파랑
                        "bottom": "👖 rgba(165, 214, 167, 0.1)", # 부드러운 초록
                        "onepiece": "👗 rgba(186, 104, 200, 0.1)" # 부드러운 보라
                    }
                    color = category_colors.get(category.lower(), "rgba(158, 158, 158, 0.1)")
                    
                    st.markdown(f"""
                    
                        <h5 style='margin: 0;'>{category} (신뢰도: {confidence:.2f})</h5>

                    """, unsafe_allow_html=True)

                    attributes_dict = json.loads(attr_json)
                    if attributes_dict:
                        attr_data = []
                        for key, value_dict in attributes_dict.items():
                            attr_data.append({
                                "속성": key,
                                "값": value_dict.get('value'),
                                "신뢰도": f"{value_dict.get('confidence'):.2f}"
                            })
                        
                        if attr_data:
                            df = pd.DataFrame(attr_data)
                            # 테이블 스타일링 - 부드러운 색상과 투명도 사용
                            st.table(df.style.set_properties(**{
                                'background-color': 'rgba(255, 255, 255, 0.05)',
                                'color': 'currentColor',  # 현재 테마의 텍스트 색상 사용
                                'border-color': 'rgba(128, 128, 128, 0.2)'
                            }))
                    else:
                        st.info("세부 속성 정보가 없습니다.")

render()