from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import time
import math
import numpy as np # เพิ่ม numpy สำหรับคำนวณแสง

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables
baseline_shoulder_width = None
is_calibrating = False
calibration_start_time = 0
calibration_duration = 3
calibration_data_x = []

breathing_state = "IDLE" 
breath_start_time = 0
INHALE_SEC = 5
HOLD_SEC = 8
EXHALE_SEC = 5

# ตัวแปรสำหรับส่งข้อความไปให้ HTML
current_calib_msg = "Waiting for camera..."
current_pose_msg = ""

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

def generate_frames():
    global baseline_shoulder_width, is_calibrating, calibration_start_time, calibration_data_x
    global breathing_state, breath_start_time, current_calib_msg, current_pose_msg
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, c = frame.shape
        current_time = time.time()

        # เช็คความสว่างของภาพ (แสงน้อย)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if not results.pose_landmarks:
            current_calib_msg = "Please step into the frame"
        else:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            current_shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            shoulder_ratio = current_shoulder_width # สัดส่วนความกว้างไหล่เทียบกับจอ (0-1)

            # [อัปเดตข้อความ Calibration ไปให้ HTML]
            if brightness < 40:
                current_calib_msg = "Low light detected"
            elif shoulder_ratio > 0.45:
                current_calib_msg = "Please move back slightly"
            elif shoulder_ratio < 0.15:
                current_calib_msg = "Please move closer"
            else:
                if is_calibrating:
                    remain = int(calibration_duration - (current_time - calibration_start_time)) + 1
                    current_calib_msg = f"Calibrating... {remain}s"
                else:
                    current_calib_msg = "All set!"

            # [ระบบ Calibration]
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

            # --- เอา mp_drawing.draw_landmarks ออกตรงนี้ ภาพจะไม่มีเส้นบนหน้าและตัวแล้ว ---

            # [ระบบเช็คท่าทาง Session]
            current_pose_msg = "" # เคลียร์ข้อความเริ่มต้น
            if breathing_state in ["INHALE", "HOLD"]:
                # คำนวณจุดเป้าหมายเหมือนเดิม แต่ไม่ต้องวาดเส้นสีน้ำเงินแล้ว
                target_wrist_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                target_wrist_y = int(left_shoulder.y * h)

                if breathing_state == "HOLD":
                    lw_px = (int(left_wrist.x * w), int(left_wrist.y * h))
                    rw_px = (int(right_wrist.x * w), int(right_wrist.y * h))
                    
                    dist_left = math.hypot(lw_px[0] - target_wrist_x, lw_px[1] - target_wrist_y)
                    dist_right = math.hypot(rw_px[0] - target_wrist_x, rw_px[1] - target_wrist_y)
                    
                    tolerance = 60
                    if dist_left < tolerance and dist_right < tolerance:
                        current_pose_msg = "PERFECT POSTURE!"
                    else:
                        current_pose_msg = "Push Arms Forward to match Blue Line" # (แก้ข้อความเป็นคำแนะนำอื่นได้ตามชอบนะครับ)

        # [ระบบจับเวลาลมหายใจของฝั่ง Python]
        if breathing_state != "IDLE":
            elapsed_breath = current_time - breath_start_time
            if breathing_state == "INHALE" and elapsed_breath > INHALE_SEC:
                breathing_state = "HOLD"
                breath_start_time = current_time
            elif breathing_state == "HOLD" and elapsed_breath > HOLD_SEC:
                breathing_state = "EXHALE"
                breath_start_time = current_time
            elif breathing_state == "EXHALE" and elapsed_breath > EXHALE_SEC:
                breathing_state = "INHALE" # วนลูป
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

# [NEW] Endpoint ส่งข้อความสถานะให้ HTML ดึงไปโชว์
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