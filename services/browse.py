import streamlit as st
import os

def render():
    st.title("이미지 탐색")
    
    upload_folder = "assets/uploads/"
    
    if not os.path.exists(upload_folder):
        st.warning("아직 업로드된 이미지가 없습니다.")
    else:
        image_files = os.listdir(upload_folder)
        if image_files:
            for img_file in image_files:
                img_path = os.path.join(upload_folder, img_file)
                st.image(img_path, caption=img_file, use_column_width=True)
        else:
            st.warning("아직 업로드된 이미지가 없습니다.")