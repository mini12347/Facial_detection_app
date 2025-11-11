import cv2
import streamlit as st
from PIL import Image
import io
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(scaleFactor, minNeighbors, color):
    stframe = st.empty()  
    cap = cv2.VideoCapture(0)
    st.info("Press **Stop** below to quit.")
    stop_btn = st.button("🛑 Stop Detection", key="stop_btn")
    snapshot_btn = st.button("📸 Take Snapshot", key="snapshot_btn")
    snapshot = None 

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to capture video frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        
        hex_color = color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        if snapshot_btn:
            snapshot = frame_rgb.copy()
            st.success("📸 Snapshot taken!")
            cv2.imwrite('try.jpg',frame)
        if stop_btn:
            break

    cap.release()    


def app():
    st.set_page_config(page_title="Face Detection App", page_icon="👤", layout="centered")

    st.markdown("""
        <h1 style="text-align:center; color:#4CAF50;">👁️ Face Detection using Viola–Jones Algorithm</h1>
        <p style="text-align:center;">Detect faces in real time using your webcam.</p>
        <hr style="border:1px solid #4CAF50;">
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Detection Settings")
    scaleFactor = st.sidebar.slider("Scale Factor", 1.05, 2.0, 1.3, 0.05)
    minNeighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5)
    color = st.sidebar.color_picker("Rectangle Color", "#00FF00")

    st.write("Press the button below to start detecting faces 👇")
    if st.button("🎥 Start Face Detection"):
        detect_faces(scaleFactor, minNeighbors, color)

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color:white;
            border-radius:10px;
            height:50px;
            width:100%;
            font-size:18px;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
