from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import threading

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

current_calib_msg = "Waiting for camera..."
current_pose_msg = ""

# เวลาตรวจจับสำหรับการทดสอบ (ปรับให้น้อยลงเพื่อให้เห็นผลไวขึ้น)
FHP_TIME_LIMIT = 2.0       
ROUNDED_TIME_LIMIT = 2.0   
STATIC_TIME_LIMIT = 5.0   

fhp_start_time = None
rounded_start_time = None
static_start_time = None
static_anchor = None

tracking_active = False
manual_session = False
camera_thread = None
output_frame = None

# ---------------------------------------------------------
# Helper Function: คำนวณมุมระหว่าง 3 จุด
# ---------------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def tracking_loop():
    global tracking_active, output_frame
    global baseline_shoulder_width, is_calibrating, calibration_start_time, calibration_data_x
    global breathing_state, breath_start_time, current_calib_msg, current_pose_msg
    global fhp_start_time, rounded_start_time, static_start_time, static_anchor
    
    # ใช้ MJPG เพื่อให้กล้องโหลดไว ไม่ค้าง 20 วินาที
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while tracking_active:
            ret, frame = cap.read()
            
            if not ret: 
                time.sleep(0.1)
                continue

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
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                
                current_shoulder_width = abs(left_shoulder.x - right_shoulder.x)

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

                for i in range(11):
                    results.pose_landmarks.landmark[i].visibility = 0

                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    BODY_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=2)
                )

                if baseline_shoulder_width and not is_calibrating:
                    
                    if breathing_state == "IDLE":
                        # ========================================================
                        # โหมดที่ 1: ตรวจจับท่าทางออฟฟิศซินโดรม (Monitoring Mode)
                        # ========================================================
                        active_warnings = []

                        ear_y = (left_ear.y + right_ear.y) / 2.0
                        ear_z = (left_ear.z + right_ear.z) / 2.0
                        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
                        shoulder_z = (left_shoulder.z + right_shoulder.z) / 2.0
                        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0

                        # 1. ตรวจจับยื่นหน้า (Forward Head)
                        delta_y = shoulder_y - ear_y 
                        delta_z = shoulder_z - ear_z 

                        cva_angle = 90.0 
                        if delta_z > 0: 
                            cva_angle = math.degrees(math.atan2(delta_y, delta_z))

                        # ปรับความเซนซิทีฟให้จับง่ายขึ้น
                        if cva_angle < 65.0:
                            if fhp_start_time is None: 
                                fhp_start_time = current_time
                            elif (current_time - fhp_start_time) >= FHP_TIME_LIMIT:
                                active_warnings.append("Forward Head")
                        else:
                            fhp_start_time = None 

                        # 2. ตรวจจับไหล่ห่อ (Rounded Shoulders)
                        if current_shoulder_width < (baseline_shoulder_width * 0.95):
                            if rounded_start_time is None: 
                                rounded_start_time = current_time
                            elif (current_time - rounded_start_time) >= ROUNDED_TIME_LIMIT:
                                active_warnings.append("Rounded Shoulders")
                        else:
                            rounded_start_time = None

                        # 3. ตรวจจับการนั่งแช่ (Prolonged Static Posture)
                        current_coords = (ear_y, shoulder_y, shoulder_x)
                        threshold_dist = baseline_shoulder_width * 0.08 

                        if static_anchor is None:
                            static_anchor = current_coords
                            static_start_time = current_time
                        else:
                            dist = math.sqrt(
                                (current_coords[0] - static_anchor[0])**2 +
                                (current_coords[1] - static_anchor[1])**2 +
                                (current_coords[2] - static_anchor[2])**2
                            )
                            
                            if dist > threshold_dist:
                                static_anchor = current_coords
                                static_start_time = current_time
                            elif (current_time - static_start_time) >= STATIC_TIME_LIMIT:
                                active_warnings.append("Static Posture")

                        if active_warnings:
                            current_pose_msg = " | ".join(active_warnings)
                        else:
                            current_pose_msg = ""
                            
                    else:
                        # ========================================================
                        # โหมดที่ 2: ตรวจจับองศาการทำกายภาพบำบัด (Exercise Mode)
                        # ========================================================
                        
                        core_visibility = min(nose.visibility, left_shoulder.visibility, right_shoulder.visibility)
                        
                        if core_visibility < 0.5:
                            current_pose_msg = "PLEASE STAY IN FRAME"
                        else:
                            # --- 1. เกณฑ์การประเมิน ท่าดัดตนแก้เกียจ ---
                            l_angle = calculate_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y])
                            r_angle = calculate_angle([right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y], [right_wrist.x, right_wrist.y])
                            
                            p1_arms_extended = (l_angle >= 125) and (r_angle >= 125)
                            
                            p1_l_wrist_high = (left_wrist.y < nose.y) or (left_wrist.y < 0.1)
                            p1_r_wrist_high = (right_wrist.y < nose.y) or (right_wrist.y < 0.1)
                            
                            z_tolerance = 0.1
                            p1_elbows_retracted = (left_elbow.z > left_shoulder.z - z_tolerance) and \
                                                  (right_elbow.z > right_shoulder.z - z_tolerance)
                            
                            is_pose_1 = p1_arms_extended and (p1_l_wrist_high and p1_r_wrist_high) and p1_elbows_retracted

                            # --- 2. เกณฑ์การประเมิน ท่าดัดตนบิดลำตัว ---
                            dx = abs(right_shoulder.x - left_shoulder.x)
                            dz = abs(right_shoulder.z - left_shoulder.z)
                            shoulder_twist_angle = math.degrees(math.atan2(dz, dx + 1e-6))
                            
                            p2_is_twisted = shoulder_twist_angle >= 12
                            
                            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
                            if right_shoulder.z > left_shoulder.z: 
                                p2_head_turned = nose.x > shoulder_center_x
                            else: 
                                p2_head_turned = nose.x < shoulder_center_x
                                
                            chest_y = (left_shoulder.y + right_shoulder.y) / 2.0
                            
                            p2_hand_placed = (left_wrist.y > chest_y or left_wrist.y > 0.9) or \
                                             (right_wrist.y > chest_y or right_wrist.y > 0.9)

                            is_pose_2 = p2_is_twisted and p2_head_turned and p2_hand_placed

                            if is_pose_1 or is_pose_2:
                                current_pose_msg = "PERFECT POSTURE!"
                            else:
                                current_pose_msg = "PLEASE ADJUST YOUR POSTURE"

            # การควบคุมสเตทการหายใจ
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
            if ret:
                output_frame = buffer.tobytes()

    finally:
        cap.release()

def generate_frames():
    global tracking_active, camera_thread, output_frame, manual_session
    
    started_by_feed = False
    if not tracking_active:
        tracking_active = True
        started_by_feed = True
        camera_thread = threading.Thread(target=tracking_loop, daemon=True)
        camera_thread.start()

    try:
        while tracking_active:
            if output_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
            time.sleep(0.016)
    finally:
        if started_by_feed and not manual_session:
            tracking_active = False

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

@app.route('/api/session_status', methods=['GET'])
def session_status():
    global manual_session
    return jsonify({"active": manual_session})

@app.route('/api/toggle_session', methods=['POST'])
def toggle_session():
    global tracking_active, camera_thread, manual_session
    data = request.json or {}
    action = data.get('action')

    if action == 'start':
        manual_session = True
        if not tracking_active:
            tracking_active = True
            camera_thread = threading.Thread(target=tracking_loop, daemon=True)
            camera_thread.start()
        return jsonify({"status": "started"})
        
    elif action == 'stop':
        manual_session = False
        tracking_active = False
        return jsonify({"status": "stopped"})
        
    return jsonify({"status": "error"})

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