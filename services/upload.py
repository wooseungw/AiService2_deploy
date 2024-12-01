import streamlit as st
import os
from db import add_user_image, get_user_images, delete_user_image, add_image_attributes
from PIL import Image
import uuid
import torch
from ultralytics import YOLO
from yolo_w_multi_classhead_vis import YOLO_MultiClass, process_labels_to_list, category_encodings, attribute_translation, category_translation
from torchvision import transforms
from fashionDetector import FashionDetector

# 카테고리별 인코딩
categories = {
    "top": 0, "blouse": 1, "casual_top": 2, "knitwear": 3, "shirt": 4, "vest": 5,
    "coat": 6, "jacket": 7, "jumper": 8, "padding": 9, "jeans": 10, "pants": 11,
    "skirt": 12, "leggings": 13, "dress": 14, "jumpsuit": 15, "swimwear": 16
}

category_encodings = {
    "length": {"short": 0, "midi": 1, "long": 2},
    "color": {
        "black": 0, "white": 1, "gray": 2, "red": 3, "pink": 4, "orange": 5, "beige": 6,
        "brown": 7, "yellow": 8, "green": 9, "khaki": 10, "mint": 11, "blue": 12,
        "navy": 13, "skyblue": 14, "purple": 15, "lavender": 16, "wine": 17,
        "neon": 18, "gold": 19
    },
    "material": {
        "fur": 0, "knit": 1, "mustang": 2, "lace": 3, "suede": 4, "linen": 5,
        "angora": 6, "mesh": 7, "corduroy": 8, "fleece": 9, "sequin": 10, "neoprene": 11,
        "denim": 12, "silk": 13, "jersey": 14, "spandex": 15, "tweed": 16,
        "jacquard": 17, "velvet": 18, "leather": 19, "vinyl": 20, "cotton": 21,
        "wool": 22, "chiffon": 23, "synthetic": 24
    },
    "sleeve_length": {"sleeveless": 0, "3/4_sleeve": 1, "short_sleeve": 2, "long_sleeve": 3, "cap": 4},
    "neckline": {
        "round": 0, "square": 1, "u_neck": 2, "collarless": 3, "v_neck": 4,
        "hood": 5, "halter": 6, "turtleneck": 7, "offshoulder": 8,
        "boatneck": 9, "one_shoulder": 10, "sweetheart": 11
    },
    "fit": {"normal": 0, "skinny": 1, "loose": 2, "wide": 3, "oversized": 4, "tight": 5}
}


# 이미지 리사이즈 및 저장 함수
def save_and_resize_image(uploaded_file, output_size=(640, 640)):
    unique_filename = f"{uuid.uuid4()}.jpeg"
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    save_path = os.path.join(upload_dir, unique_filename)
    image = Image.open(uploaded_file)
    
    # 이미지가 RGBA 모드일 경우 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    resized_image = image.resize(output_size)
    resized_image.save(save_path, format="JPEG")  # JPEG 포맷으로 저장
    
    return unique_filename, save_path  # 이미지 파일 이름과 경로 반환
    
def render():
    st.title("이미지 올리기")
    
    user_id = st.session_state.get('id')
    if not user_id:
        st.error("로그인이 필요합니다.")
        st.stop()
    else:
        user_id = int(user_id)  # 정수형으로 변환
        
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img_name, img_url = save_and_resize_image(uploaded_file)
        st.success(f"이미지가 업로드되었습니다: {img_url}")
        
        # 데이터베이스에 이미지 정보 추가 후 이미지 ID 가져오기
        image_id = add_user_image(user_id, img_name, img_url)
        
        # FashionDetector 초기화
        detection_model_path = "./runs/detect/l_640_dropout025_more_cat_3/weights/best.pt"
        attr_model_path = "yolo11l.pt"
        attr_weights_path = "./fashion_classification_l/best_model.pt"

        detector = FashionDetector(
            detection_model_path=detection_model_path,
            attr_model_path=attr_model_path,
            attr_weights_path=attr_weights_path,
            categories=categories,
            category_encodings=category_encodings
        )
        
        # 이미지 처리 및 결과 얻기
        results = detector.process(img_url)
        
        # 결과를 데이터베이스에 저장
        all_attributes = []
        for category in ['outer', 'top', 'bottom', 'onepiece']:
            for item in results.get(category, []):
                item_data = {
                    'category': item['category'],
                    'bounding_box': item['bounding_box'],
                    'confidence': item['confidence'],
                    'attributes': item['attributes']
                }
                all_attributes.append(item_data)
        
        add_image_attributes(image_id, all_attributes)
        
        st.success("FashionDetector 결과가 저장되었습니다.")
        

    # 특정 사용자가 업로드한 이미지 표시
    user_images = get_user_images(user_id)
    st.subheader("업로드한 이미지 갤러리")

    # 이미지를 3열 갤러리 형태로 나열
    num_columns = 3
    for i in range(0, len(user_images), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(user_images):
                # 튜플의 네 가지 요소를 모두 받습니다
                image_id, filename, filepath, upload_date = user_images[i + j]
                with col:
                    st.image(filepath, caption=f"{filename[:10]}... ({upload_date})", use_container_width=True)
                    if st.button("삭제", key=f"delete_{filename}"):
                        if delete_user_image(user_id, image_id):
                            st.success("이미지가 삭제되었습니다.")
                            st.rerun()  # 페이지 새로고침
                        else:
                            st.error("이미지 삭제에 실패했습니다.")
                            
render()
