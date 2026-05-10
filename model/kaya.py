from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import time
import math
import numpy as np

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# สร้างเซ็ตของการเชื่อมต่อเฉพาะช่วงลำตัว
BODY_CONNECTIONS = frozenset([conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] > 10 and conn[1] > 10])

# Variables สำหรับ Calibration
baseline_shoulder_width = None
is_calibrating = False
calibration_start_time = 0
calibration_duration = 3
calibration_data_x = []

# Variables สำหรับลมหายใจ
breathing_state = "IDLE" 
breath_start_time = 0
INHALE_SEC = 5
HOLD_SEC = 8
EXHALE_SEC = 5

current_calib_msg = "Waiting for camera..."
current_pose_msg = ""

# ---------------------------------------------------------
# Variables สำหรับ Medical & Ergonomic Thresholds
# ---------------------------------------------------------
# ตั้งเวลาเป็นวินาที (ของจริง: 5 นาที = 300, 45 นาที = 2700)
# แนะนำ: ตอนทดสอบรันโปรแกรม ให้เปลี่ยนเป็น 5 และ 10 วินาทีดูครับ จะได้ไม่ต้องรอนาน
FHP_TIME_LIMIT = 5       
ROUNDED_TIME_LIMIT = 5   
STATIC_TIME_LIMIT = 10   

fhp_start_time = None
rounded_start_time = None
static_start_time = None
static_anchor = None

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def generate_frames():
    global baseline_shoulder_width, is_calibrating, calibration_start_time, calibration_data_x
    global breathing_state, breath_start_time, current_calib_msg, current_pose_msg
    global fhp_start_time, rounded_start_time, static_start_time, static_anchor
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, c = frame.shape
        current_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if not results.pose_landmarks:
            current_calib_msg = "Please step into the frame"
        else:
            landmarks = results.pose_landmarks.landmark
            
            # ดึงจุดอ้างอิงที่จำเป็น
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            
            current_shoulder_width = abs(left_shoulder.x - right_shoulder.x)

            # [ระบบ Calibration UI]
            if brightness < 40:
                current_calib_msg = "Low light detected"
            elif current_shoulder_width > 0.45:
                current_calib_msg = "Please move back slightly"
            elif current_shoulder_width < 0.15:
                current_calib_msg = "Please move closer"
            else:
                if is_calibrating:
                    remain = int(calibration_duration - (current_time - calibration_start_time)) + 1
                    current_calib_msg = f"Calibrating... {remain}s"
                else:
                    current_calib_msg = "All set!"

            # [ระบบคำนวณ Baseline ตอนเริ่ม]
            if is_calibrating:
                elapsed_time = current_time - calibration_start_time
                if elapsed_time < calibration_duration:
                    calibration_data_x.append(current_shoulder_width)
                else:
                    if len(calibration_data_x) > 0:
                        baseline_shoulder_width = sum(calibration_data_x) / len(calibration_data_x)
                    is_calibrating = False
                    calibration_data_x = []
                    current_calib_msg = "Calibration Complete"

            # ซ่อนใบหน้าและวาดเส้น
            for i in range(11):
                results.pose_landmarks.landmark[i].visibility = 0

            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                BODY_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=2)
            )

            # ---------------------------------------------------------
            # AI Posture Rules (Medical Standard)
            # ---------------------------------------------------------
            if baseline_shoulder_width and not is_calibrating:
                active_warnings = []

                # หาจุดศูนย์กลางของหูและไหล่ เพื่อใช้คำนวณ
                ear_y = (left_ear.y + right_ear.y) / 2.0
                ear_z = (left_ear.z + right_ear.z) / 2.0
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
                shoulder_z = (left_shoulder.z + right_shoulder.z) / 2.0
                shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0

                # 1. Forward Head Posture (CVA < 45 องศา เป็นเวลา 5 นาที)
                # คำนวณความต่างของแกน Y (ความสูง) และแกน Z (ความลึก) เพื่อหามุม
                delta_y = shoulder_y - ear_y 
                delta_z = shoulder_z - ear_z 

                cva_angle = 90.0 # ค่าเริ่มต้นนั่งตรง
                if delta_z > 0: # ถ้าหูยื่นมาใกล้กล้องมากกว่าไหล่
                    cva_angle = math.degrees(math.atan2(delta_y, delta_z))

                if cva_angle < 45.0:
                    if fhp_start_time is None: 
                        fhp_start_time = current_time
                    elif (current_time - fhp_start_time) >= FHP_TIME_LIMIT:
                        active_warnings.append("Forward Head Detected")
                else:
                    fhp_start_time = None # รีเซ็ตเวลาถ้ากลับมานั่งตรง

                # 2. Rounded Shoulders (ความกว้าง < 90% ของ Baseline เป็นเวลา 5 นาที)
                if current_shoulder_width < (baseline_shoulder_width * 0.90):
                    if rounded_start_time is None: 
                        rounded_start_time = current_time
                    elif (current_time - rounded_start_time) >= ROUNDED_TIME_LIMIT:
                        active_warnings.append("Rounded Shoulders Detected")
                else:
                    rounded_start_time = None

                # 3. Prolonged Static Posture (ไม่ขยับ > 5cm เป็นเวลา 45 นาที)
                # แปลง 5 ซม. เป็นสัดส่วนบนจอ (ประมาณ 12.5% ของความกว้างไหล่มาตรฐาน)
                current_coords = (ear_y, shoulder_y, shoulder_x)
                threshold_dist = baseline_shoulder_width * 0.125

                if static_anchor is None:
                    static_anchor = current_coords
                    static_start_time = current_time
                else:
                    # คำนวณระยะกระจัด 3 มิติ ว่าขยับไปจากจุด Anchor เริ่มต้นเท่าไหร่
                    dist = math.sqrt(
                        (current_coords[0] - static_anchor[0])**2 +
                        (current_coords[1] - static_anchor[1])**2 +
                        (current_coords[2] - static_anchor[2])**2
                    )
                    
                    if dist > threshold_dist:
                        # ขยับตัวเกิน 5 ซม. แล้ว -> รีเซ็ตจุด Anchor และเวลาใหม่
                        static_anchor = current_coords
                        static_start_time = current_time
                    elif (current_time - static_start_time) >= STATIC_TIME_LIMIT:
                        active_warnings.append("Prolonged Static Posture: Time to stretch!")

                # อัปเดตข้อความส่งไปที่ HTML
                if active_warnings:
                    current_pose_msg = " | ".join(active_warnings)
                else:
                    current_pose_msg = ""


        # [ระบบวงล้อลมหายใจ]
        if breathing_state != "IDLE":
            elapsed_breath = current_time - breath_start_time
            if breathing_state == "INHALE" and elapsed_breath > INHALE_SEC:
                breathing_state = "HOLD"
                breath_start_time = current_time
            elif breathing_state == "HOLD" and elapsed_breath > HOLD_SEC:
                breathing_state = "EXHALE"
                breath_start_time = current_time
            elif breathing_state == "EXHALE" and elapsed_breath > EXHALE_SEC:
                breathing_state = "INHALE"
                breath_start_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    global current_calib_msg, current_pose_msg
    return jsonify({
        "calib_msg": current_calib_msg,
        "pose_msg": current_pose_msg
    })

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global is_calibrating, calibration_start_time, calibration_data_x, baseline_shoulder_width
    if not is_calibrating:
        is_calibrating = True
        calibration_start_time = time.time()
        calibration_data_x.clear()
        baseline_shoulder_width = None
        return jsonify({"status": "success"})
    return jsonify({"status": "ignored"})

@app.route('/api/start_pose', methods=['POST'])
def start_pose():
    global breathing_state, breath_start_time
    breathing_state = "INHALE"
    breath_start_time = time.time()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    print(">> Backend Server Running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)