# app.py
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import pygame
import time
import os
import pyttsx3
import threading
import easyocr
from scipy.spatial import distance as dist

# === VOICE ALERT SETUP (FIXED) ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
time.sleep(0.5)

def speak(text):
    def run():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run, daemon=True).start()

# === TURN DETECTION (LANE CURVATURE) - FIXED: NO FALSE ALERTS ===
def detect_curvature(frame):
    h, w = frame.shape[:2]
    
    center_x_start = int(w * 0.25)
    center_x_end = int(w * 0.75)
    center_y_start = int(h * 0.4)
    center_y_end = h
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[center_y_start:center_y_end, center_x_start:center_x_end] = 255
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=60,
                            minLineLength=100, maxLineGap=30)
    if lines is None or len(lines) < 4:
        return None
    
    left, right = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x1 < center_x_start or x2 < center_x_start or x1 > center_x_end or x2 > center_x_end:
            continue
        if y1 < center_y_start or y2 < center_y_start:
            continue
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
        if slope < -0.6:
            left.append(slope)
        elif slope > 0.6:
            right.append(slope)
    
    if len(left) >= 3 and len(right) >= 3:
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        diff = abs(left_mean - right_mean)
        if diff > 1.2 and abs(left_mean) > 0.8 and abs(right_mean) > 0.8:
            return "SHARP TURN AHEAD!"
    return None

# === SPEED LIMIT DETECTION (OCR) ===
reader = easyocr.Reader(['en'])
def detect_speed_limit(frame, boxes):
    for box in boxes:
        if int(box.cls) == 10:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            result = reader.readtext(cropped)
            for (_, text, prob) in result:
                if prob > 0.6 and any(c.isdigit() for c in text):
                    return text.strip()
    return None

# === TRAFFIC LIGHT DETECTION ===
def detect_traffic_light(frame, boxes):
    for box in boxes:
        if int(box.cls) == 9:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            light = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
            red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            green = cv2.inRange(hsv, (40, 70, 50), (80, 255, 255))
            red_pixels = cv2.countNonZero(red1 + red2)
            green_pixels = cv2.countNonZero(green)
            if red_pixels > 150 and red_pixels > green_pixels:
                return "RED LIGHT!"
            elif green_pixels > 150 and green_pixels > red_pixels:
                return "GREEN LIGHT!"
    return None

# === SESSION STATE (REMOVED distraction/drowsiness) ===
for key in ["front_url", "back_url", "front_ip", "back_ip", "stop", "last_time", "demo_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None if "ip" not in key and "url" not in key else "192.168.1.100"
if "prev_centers" not in st.session_state:
    st.session_state.prev_centers = {}
if "prev_sizes" not in st.session_state:
    st.session_state.prev_sizes = {}
if "streaming" not in st.session_state:
    st.session_state.streaming = True
if "stop" not in st.session_state:
    st.session_state.stop = False
if "last_highspeed_alert" not in st.session_state:
    st.session_state.last_highspeed_alert = 0
if "last_blind_alert" not in st.session_state:
    st.session_state.last_blind_alert = 0
if "last_turn_alert" not in st.session_state:
    st.session_state.last_turn_alert = 0
if "last_speed_limit_alert" not in st.session_state:
    st.session_state.last_speed_limit_alert = 0
if "last_traffic_light_alert" not in st.session_state:
    st.session_state.last_traffic_light_alert = 0

# === SETUP ===
st.set_page_config(page_title="SpeedGuard", page_icon="Car", layout="wide")
st.title("SpeedGuard: AI Dashcam Safety System")
st.markdown("**High-Speed, Blind Spot, Turn, Speed Limit, Traffic Light Alerts**")

# === FIXED PYGAME INIT (CORRECT SYNTAX) ===
try:
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
except:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

beep_path = "beep.wav"
if not os.path.exists(beep_path):
    st.error("`beep.wav` not found! Run `python generate_beep.py` first.")
    st.stop()
beep = pygame.mixer.Sound(beep_path)
beep.set_volume(1.0)

# === FIXED YOLO LOADING (NO .bn ERROR) ===
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    model.fuse = lambda *args, **kwargs: model  # DISABLE FUSE
    return model

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

# === DEMO MODE: UPLOAD VIDEOS ===
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

# === DEMO MODE: PRE-LOADED VIDEOS FROM GOOGLE DRIVE ===
else:
    st.info("**Pre-loaded Demo** – Loading videos from Google Drive for demo ... Dont close the app!, wait for 2 mins")

    import gdown

    FRONT_ID = "1wZcagbZnvPyAeyse0CSP75KIySCHj2wA"  # front.mp4
    BACK_ID  = "1SxKs_F6V2FmK2qoEJxsVyvsqfGqzZkvQ"   # back.mp4

    front_path = "temp_front_drive.mp4"
    back_path  = "temp_back_drive.mp4"

    if not os.path.exists(front_path):
        st.warning("Downloading front video from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FRONT_ID}", front_path, quiet=False)
    if not os.path.exists(back_path):
        st.warning("Downloading back video from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={BACK_ID}", back_path, quiet=False)

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        st.error("Failed to download videos from Google Drive!")
        st.stop()

    cap_front = cv2.VideoCapture(front_path)
    cap_back = cv2.VideoCapture(back_path)
    st.success("Google Drive demo videos loaded!")

# === VALIDATE CAPTURES ===
if not cap_front.isOpened() or not cap_back.isOpened():
    st.error("Failed to open one or both video streams.")
    st.stop()

# === TEST VOICE + BEEP ===
col_beep1, col_beep2 = st.columns([1, 3])
with col_beep1:
    if st.button("TEST ALERT", type="secondary"):
        beep.play()
        speak("SpeedGuard alert test!")
        st.toast("Alert played!")
with col_beep2:
    force_beep = st.checkbox("FORCE ALERT (Every 5 sec)", value=False)

# === DISPLAY PLACEHOLDERS ===
frame_ph = st.empty()
alert_ph = st.empty()

# === AUTO-START + STOP BUTTON ===
if st.session_state.streaming:
    if st.button("STOP STREAM", type="primary"):
        st.session_state.streaming = False
        st.session_state.stop = True
        st.rerun()
else:
    if st.button("START STREAM", type="primary"):
        st.session_state.streaming = True
        st.session_state.stop = False
        st.rerun()

# === MAIN PROCESSING LOOP ===
if st.session_state.streaming:
    last_beep_time = time.time()
    frame_count = 0

    while st.session_state.streaming:
        ret1, frame1 = cap_front.read()
        ret2, frame2 = cap_back.read()
        if not ret1 or not ret2:
            cap_front.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_back.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame1 = cv2.resize(frame1, (640, 360))
        frame2 = cv2.resize(frame2, (640, 360))

        current_time = time.time()
        fps = 1 / (current_time - st.session_state.get("last_time", current_time)) if frame_count > 0 else 0
        st.session_state.last_time = current_time
        cv2.putText(frame1, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

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
        if highspeed_detected and (current_time - st.session_state.last_highspeed_alert > 5.0):
            alert = "HIGH-SPEED VEHICLE APPROACHING!"
            beep.play()
            speak("High speed vehicle approaching!")
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
            side = "left" if blind_left else "right"
            alert = f"BLIND SPOT {side.upper()}!"
            beep.play()
            speak(f"Blind spot {side}!")
            st.session_state.last_blind_alert = current_time

        # === TURN DETECTION (FRONT CAM ONLY) ===
        turn_alert = detect_curvature(frame1)
        if turn_alert and (current_time - st.session_state.last_turn_alert > 8.0):
            alert = turn_alert
            beep.play()
            speak("Sharp turn ahead! Slow down!")
            cv2.putText(frame1, "TURN AHEAD!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            st.session_state.last_turn_alert = current_time

        # === SPEED LIMIT DETECTION (FRONT CAM) ===
        speed_limit = detect_speed_limit(frame1, results1.boxes)
        if speed_limit and (current_time - st.session_state.last_speed_limit_alert > 10.0):
            alert = f"SPEED LIMIT DETECTED: {speed_limit}"
            beep.play()
            speak(f"Speed limit {speed_limit}!")
            cv2.putText(frame1, f"SPEED LIMIT: {speed_limit}", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            st.session_state.last_speed_limit_alert = current_time

        # === TRAFFIC LIGHT DETECTION (FRONT CAM) ===
        traffic_light = detect_traffic_light(frame1, results1.boxes)
        if traffic_light and (current_time - st.session_state.last_traffic_light_alert > 5.0):
            alert = traffic_light
            beep.play()
            speak(traffic_light.replace("!", ""))
            cv2.putText(frame1, traffic_light, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            st.session_state.last_traffic_light_alert = current_time

        # === FORCE ALERT ===
        if force_beep and (time.time() - last_beep_time > 5):
            alert = "FORCE ALERT: OK"
            beep.play()
            speak("Test alert activated!")
            last_beep_time = time.time()

        st.session_state.prev_centers = current_centers.copy()
        st.session_state.prev_sizes = current_sizes.copy()

        with frame_ph.container():
            c1, c2 = st.columns(2)
            c1.image(frame1, channels="BGR", caption="FRONT CAM (Driver + Road)")
            c2.image(frame2, channels="BGR", caption="BACK CAM (Speeding Vehicles)")

        alert_ph.empty()
        if alert:
            alert_ph.error(f"**{alert}**")
        else:
            alert_ph.success("All Clear")

        time.sleep(0.05)
        frame_count += 1

# === CLEANUP ===
if st.session_state.get("stop", False):
    if 'cap_front' in locals():
        cap_front.release()
    if 'cap_back' in locals():
        cap_back.release()
    for f in ["temp_front_drive.mp4", "temp_back_drive.mp4"]:
        if os.path.exists(f):
            os.remove(f)
    if mode.startswith("Demo: Upload"):
        for f in ["temp_front.mp4", "temp_back.mp4"]:
            if os.path.exists(f):
                os.remove(f)
    st.success("Stream stopped. Click **START STREAM** to resume.")
