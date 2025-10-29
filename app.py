# app.py
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time
import os

# === SESSION STATE ===
for key in ["front_url", "back_url", "front_ip", "back_ip", "stop", "last_time", "demo_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None if "ip" not in key and "url" not in key else "192.168.1.100"
if "prev_centers" not in st.session_state:
    st.session_state.prev_centers = {}
if "prev_sizes" not in st.session_state:
    st.session_state.prev_sizes = {}
if "last_highspeed_alert" not in st.session_state:
    st.session_state.last_highspeed_alert = 0
if "last_blind_alert" not in st.session_state:
    st.session_state.last_blind_alert = 0

# === SETUP ===
st.set_page_config(page_title="SpeedGuard", page_icon="Car", layout="wide")
st.title("SpeedGuard: AI Dashcam Safety System")
st.markdown("**High-Speed Incoming + Blind Spot Alerts**")

# === BEEP USING st.audio (HIDDEN + AUTOPLAY) ===
beep_path = "beep.wav"
if not os.path.exists(beep_path):
    st.error("`beep.wav` not found! Run `python generate_beep.py` first.")
    st.stop()

# Load YOLO
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')
model = load_model()

# === MODE SELECTION ===
st.markdown("### Select Input Mode")
mode = st.radio(
    "Choose input source",
    ["Live: Laptop + Phone (Back)", "Demo: Upload Videos", "Demo: Pre-loaded Videos"],
    index=2,
    horizontal=True
)

# === LIVE MODE ===
if mode == "Live: Laptop + Phone (Back)":
    st.info("**Live Mode**: Phone = Front Cam | Laptop = Back Cam (via DroidCam)")
    st.info("1. Open [DroidCam](https://www.dev47apps.com/) on phone to Start Camera to Note IP\n"
            "2. Enter IP below and click **CONNECT**")
    col1, col2 = st.columns([3, 1])
    with col1:
        phone_ip = st.text_input("Phone IP (Front Cam)", value=st.session_state.front_ip, key="live_ip")
    with col2:
        connect_btn = st.button("CONNECT", type="primary", use_container_width=True)
    if connect_btn:
        st.session_state.front_ip = phone_ip
        st.session_state.front_url = f"http://{phone_ip}:4747/video"
        st.success(f"Connected: {phone_ip}")
        st.rerun()
    if not st.session_state.front_url:
        st.warning("Enter IP and click CONNECT")
        st.stop()
    cap_front = cv2.VideoCapture(st.session_state.front_url)
    cap_back = cv2.VideoCapture(0)

# === UPLOAD MODE ===
elif mode == "Demo: Upload Videos":
    st.info("**Upload Test Videos** to Test with your own speeding vehicle footage")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        front_file = st.file_uploader("Upload **Front** Video (MP4)", type=["mp4"])
    with col_up2:
        back_file = st.file_uploader("Upload **Back** Video (MP4)", type=["mp4"])
    if front_file and back_file:
        front_path = "temp_front.mp4"
        back_path = "temp_back.mp4"
        with open(front_path, "wb") as f:
            f.write(front_file.getbuffer())
        with open(back_path, "wb") as f:
            f.write(back_file.getbuffer())
        cap_front = cv2.VideoCapture(front_path)
        cap_back = cv2.VideoCapture(back_path)
        st.success("Videos loaded! Processing...")
    else:
        st.warning("Upload both videos to start")
        st.stop()

# === PRE-LOADED DEMO ===
else:
    st.info("**Pre-loaded Demo** to Test with built-in speeding vehicle video")
    front_path = "videos/front.mp4"
    back_path = "videos/back.mp4"
    if not os.path.exists(front_path) or not os.path.exists(back_path):
        st.error("Demo videos not found in `videos/` folder!")
        st.stop()
    cap_front = cv2.VideoCapture(front_path)
    cap_back = cv2.VideoCapture(back_path)
    st.success("Pre-loaded demo videos loaded!")

# === VALIDATE ===
if not cap_front.isOpened() or not cap_back.isOpened():
    st.error("Failed to open video streams.")
    st.stop()

# === TEST BEEP & FORCE BEEP ===
col_beep1, col_beep2 = st.columns([1, 3])
with col_beep1:
    if st.button("TEST BEEP", type="secondary"):
        with open(beep_path, "rb") as f:
            st.empty().audio(f, format="audio/wav", autoplay=True)
        st.toast("Beep played!")
with col_beep2:
    force_beep = st.checkbox("FORCE BEEP (Every 5 sec)", value=False)

# === DISPLAY ===
frame_ph = st.empty()
alert_ph = st.empty()

# === MAIN LOOP ===
stop_button = st.button("STOP", type="secondary")
if stop_button:
    st.session_state.stop = True

last_beep_time = time.time()
frame_count = 0

while not st.session_state.get("stop", False):
    ret1, frame1 = cap_front.read()
    ret2, frame2 = cap_back.read()
    if not ret1 or not ret2:
        st.warning("Video ended.")
        break
    if not ret1:
        cap_front.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret2:
        cap_back.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))

    current_time = time.time()
    fps = 1 / (current_time - st.session_state.get("last_time", current_time)) if frame_count > 0 else 0
    st.session_state.last_time = current_time
    cv2.putText(frame1, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # YOLO TRACKING WITH BOTSORT
    results1 = model.track(frame1, persist=True, tracker="botsort.yaml", verbose=False)[0]
    results2 = model.track(frame2, persist=True, tracker="botsort.yaml", verbose=False)[0]

    alert = None
    current_centers = {}
    current_sizes = {}

    # === BACK CAM: HIGH-SPEED ===
    highspeed_detected = False
    for box in results2.boxes:
        if int(box.cls) != 2: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id.item()) if box.id is not None else None
        if not track_id: continue
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        area = (x2 - x1) * (y2 - y1)
        current_centers[track_id] = center
        current_sizes[track_id] = area
        prev = st.session_state.prev_centers.get(track_id)
        prev_area = st.session_state.prev_sizes.get(track_id)
        if prev and prev_area:
            dy = center[1] - prev[1]
            size_ratio = area / prev_area if prev_area > 0 else 1
            speed_score = abs(dy) * size_ratio
            if dy < -3 and size_ratio > 1.1 and center[1] > 80 and speed_score > 4:
                highspeed_detected = True
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame2, "HIGH SPEED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame2, f"Score:{speed_score:.1f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if highspeed_detected and (current_time - st.session_state.last_highspeed_alert > 5.0):
        alert = "HIGH-SPEED VEHICLE APPROACHING!"
        with open(beep_path, "rb") as f:
            st.empty().audio(f, format="audio/wav", autoplay=True)
        st.session_state.last_highspeed_alert = current_time

    # === FRONT CAM: BLIND SPOTS ===
    blind_left = blind_right = False
    h, w = frame1.shape[:2]
    left_zone = (0, int(h*0.33), int(w*0.34), h)
    right_zone = (int(w*0.66), int(h*0.33), w, h)
    for box in results1.boxes:
        if int(box.cls) != 2: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        if left_zone[0] < cx < left_zone[2]:
            blind_left = True
            cv2.putText(frame1, "BLIND LEFT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif right_zone[0] < cx < right_zone[2]:
            blind_right = True
            cv2.putText(frame1, "BLIND RIGHT!", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if (blind_left or blind_right) and (current_time - st.session_state.last_blind_alert > 5.0):
        alert = "BLIND SPOT LEFT!" if blind_left else "BLIND SPOT RIGHT!"
        with open(beep_path, "rb") as f:
            st.empty().audio(f, format="audio/wav", autoplay=True)
        st.session_state.last_blind_alert = current_time

    if force_beep and (time.time() - last_beep_time > 5):
        alert = "FORCE BEEP: OK"
        with open(beep_path, "rb") as f:
            st.empty().audio(f, format="audio/wav", autoplay=True)
        last_beep_time = time.time()

    st.session_state.prev_centers = current_centers.copy()
    st.session_state.prev_sizes = current_sizes.copy()

    with frame_ph.container():
        c1, c2 = st.columns(2)
        c1.image(frame1, channels="BGR", caption="FRONT CAM")
        c2.image(frame2, channels="BGR", caption="BACK CAM")

    alert_ph.empty()
    if alert:
        alert_ph.error(f"**{alert}**")
    else:
        alert_ph.success("All Clear")

    time.sleep(0.05)
    frame_count += 1

# === CLEANUP ===
cap_front.release()
cap_back.release()
if mode.startswith("Demo: Upload"):
    for f in ["temp_front.mp4", "temp_back.mp4"]:
        if os.path.exists(f):
            os.remove(f)
st.success("Stream stopped.")
