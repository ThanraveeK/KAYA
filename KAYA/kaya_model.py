from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import time
import math

app = Flask(__name__)
CORS(app) # อนุญาตให้หน้าเว็บ (Frontend) ยิง API เข้ามาได้

# ----------------------------------------------------
# ตัวแปร Global สำหรับเก็บสถานะของระบบ
# ----------------------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

# เปิดกล้องเตรียมไว้
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

def generate_frames():
    global baseline_shoulder_width, is_calibrating, calibration_start_time, calibration_duration, calibration_data_x
    global breathing_state, breath_start_time
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, c = frame.shape
        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            current_shoulder_width = abs(left_shoulder.x - right_shoulder.x)

            # [ระบบ Calibration]
            if is_calibrating:
                elapsed_time = current_time - calibration_start_time
                if elapsed_time < calibration_duration:
                    calibration_data_x.append(current_shoulder_width)
                    remain_time = int(calibration_duration - elapsed_time) + 1
                    cv2.putText(frame, f"Calibrating... {remain_time}s", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                else:
                    if len(calibration_data_x) > 0:
                        baseline_shoulder_width = sum(calibration_data_x) / len(calibration_data_x)
                    is_calibrating = False
                    calibration_data_x = []

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # [ระบบเช็คท่าทาง]
            if breathing_state in ["INHALE", "HOLD"]:
                target_wrist_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                target_wrist_y = int(left_shoulder.y * h)
                
                ls_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                rs_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                
                cv2.line(frame, ls_px, (target_wrist_x, target_wrist_y), (255, 0, 0), 6)
                cv2.line(frame, rs_px, (target_wrist_x, target_wrist_y), (255, 0, 0), 6)
                cv2.circle(frame, (target_wrist_x, target_wrist_y), 15, (255, 0, 0), -1)

                if breathing_state == "HOLD":
                    lw_px = (int(left_wrist.x * w), int(left_wrist.y * h))
                    rw_px = (int(right_wrist.x * w), int(right_wrist.y * h))
                    
                    dist_left = math.hypot(lw_px[0] - target_wrist_x, lw_px[1] - target_wrist_y)
                    dist_right = math.hypot(rw_px[0] - target_wrist_x, rw_px[1] - target_wrist_y)
                    
                    tolerance = 60
                    if dist_left < tolerance and dist_right < tolerance:
                        cv2.putText(frame, "PERFECT POSTURE!", (180, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                    else:
                        cv2.putText(frame, "Push Arms Forward to match Blue Line", (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # [ระบบวงล้อลมหายใจ]
        if breathing_state != "IDLE":
            elapsed_breath = current_time - breath_start_time
            
            if breathing_state == "INHALE":
                if elapsed_breath > INHALE_SEC:
                    breathing_state = "HOLD"
                    breath_start_time = current_time
                    elapsed_breath = 0
                current_duration, instruction, color = INHALE_SEC, "INHALE (Push Arms)", (0, 255, 0)
                
            elif breathing_state == "HOLD":
                if elapsed_breath > HOLD_SEC:
                    breathing_state = "EXHALE"
                    breath_start_time = current_time
                    elapsed_breath = 0
                current_duration, instruction, color = HOLD_SEC, "HOLD (Keep Still)", (0, 255, 255)
                
            elif breathing_state == "EXHALE":
                if elapsed_breath > EXHALE_SEC:
                    breathing_state = "IDLE"
                current_duration, instruction, color = EXHALE_SEC, "EXHALE (Relax)", (0, 165, 255)
                
            if breathing_state != "IDLE":
                progress = min(elapsed_breath / current_duration, 1.0)
                end_angle = int(360 * progress)
                remain_sec = int(current_duration - elapsed_breath) + 1
                center, radius = (520, 100), 50
                
                cv2.circle(frame, center, radius, (200, 200, 200), 6)
                cv2.ellipse(frame, center, (radius, radius), 270, 0, end_angle, color, 8)
                cv2.putText(frame, str(remain_sec), (center[0]-15, center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, instruction, (center[0]-100, center[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # แทนที่จะใช้ cv2.imshow ให้เข้ารหัสภาพเป็น JPEG แล้วส่งออกไป
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------

# ท่อส่งวิดีโอ (เอา URL นี้ไปใส่ใน <img src="...">)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ปุ่ม Calibrate (แทนการกดปุ่ม 'b')
@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global is_calibrating, calibration_start_time, calibration_data_x, baseline_shoulder_width
    if not is_calibrating:
        is_calibrating = True
        calibration_start_time = time.time()
        calibration_data_x.clear()
        baseline_shoulder_width = None
        return jsonify({"status": "success", "message": "Started calibrating"})
    return jsonify({"status": "ignored", "message": "Already calibrating"})

# ปุ่มเริ่มทำท่า (แทนการกดปุ่ม 't')
@app.route('/api/start_pose', methods=['POST'])
def start_pose():
    global breathing_state, breath_start_time
    if breathing_state == "IDLE":
        breathing_state = "INHALE"
        breath_start_time = time.time()
        return jsonify({"status": "success", "message": "Exercise started"})
    return jsonify({"status": "ignored", "message": "Exercise is running"})

if __name__ == '__main__':
    print(">> Backend Server Running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)