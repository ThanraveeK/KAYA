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

# สร้างเซ็ตของการเชื่อมต่อเฉพาะช่วงลำตัว (ตัดใบหน้า ID 0-10 ออก)
BODY_CONNECTIONS = frozenset([conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] > 10 and conn[1] > 10])

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

# --- ตัวแปรสำหรับ Buffer / Range ท่าทาง ---
bad_posture_frames = 0
WARNING_THRESHOLD = 15 # ต้องทำท่าผิดค้างไว้ 15 เฟรมถึงจะเตือน (กันการขยับตัวชั่วคราว)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def generate_frames():
    global baseline_shoulder_width, is_calibrating, calibration_start_time, calibration_data_x
    global breathing_state, breath_start_time, current_calib_msg, current_pose_msg
    global bad_posture_frames
    
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
            
            # จุดอ้างอิง
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            current_shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            shoulder_ratio = current_shoulder_width

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

            # ---------------------------------------------------------
            # 1. การวาดโครงกระดูก (เฉพาะส่วนตัว)
            # ---------------------------------------------------------
            # ปิด Visibility ของจุดใบหน้า (0-10) จะได้ไม่ถูกวาดออกมา
            for i in range(11):
                results.pose_landmarks.landmark[i].visibility = 0

            # วาดเส้นเฉพาะ Body Connections
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                BODY_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=2, circle_radius=3), # สีจุด
                mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=2)  # สีเส้น
            )

            # ---------------------------------------------------------
            # 2. ระบบตรวจจับท่าทาง (Posture Logic + Hysteresis Range)
            # ---------------------------------------------------------
            detected_issue = None

            if baseline_shoulder_width and not is_calibrating and breathing_state != "IDLE":
                # A. เช็คไหล่ห่อ (Rounded Shoulders)
                # ถ้ากว้างน้อยกว่า 82% ของตอน Calibrate แปลว่าเริ่มห่อไหล่ (Range: 82% ลงไป)
                current_ratio = current_shoulder_width / baseline_shoulder_width
                
                # B. เช็คคอยื่น (Forward Head Posture)
                # เอาแกน Z ของไหล่ ลบ แกน Z ของจมูก (ยิ่งบวกเยอะ แปลว่าจมูกพุ่งแซงไหล่มาเยอะ)
                avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
                head_forward_dist = avg_shoulder_z - nose.z

                # ประเมินผล (Prioritize การแก้คอยื่นก่อน เพราะอันตรายกว่า)
                if head_forward_dist > 0.12:  # ค่า Threshold แกน Z (ปรับเพิ่มลดได้ตามสภาพกล้อง)
                    detected_issue = "Adjust Neck: Forward Head Detected"
                elif current_ratio < 0.82:    # ค่า Threshold ไหล่
                    detected_issue = "Open Chest: Rounded Shoulders Detected"

            # ---------------------------------------------------------
            # 3. ระบบ Buffer กันการเตือนจุกจิก
            # ---------------------------------------------------------
            if detected_issue:
                bad_posture_frames += 1
            else:
                # ถ้าทำถูกท่า ให้ค่อยๆ ลดค่าลง (ลดทีละ 2 เพื่อให้ฟื้นตัวกลับมาสถานะปกติได้ไวขึ้น)
                bad_posture_frames = max(0, bad_posture_frames - 2)

            # ตัดสินใจส่งข้อความไปหา HTML
            if bad_posture_frames >= WARNING_THRESHOLD:
                current_pose_msg = detected_issue
            else:
                current_pose_msg = "PERFECT POSTURE!"


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