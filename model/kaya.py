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

# Variables พื้นฐาน
baseline_shoulder_width = None
is_calibrating = False
calibration_start_time = 0
calibration_duration = 3
calibration_data_x = []
current_calib_msg = "Waiting for camera..."
current_pose_msg = ""
tracking_active = False
manual_session = False
camera_thread = None
output_frame = None

# โหมดออฟฟิศซินโดรม
FHP_TIME_LIMIT = 2.0       
ROUNDED_TIME_LIMIT = 2.0   
STATIC_TIME_LIMIT = 5.0   
fhp_start_time = None
rounded_start_time = None
static_start_time = None
static_anchor = None

# โหมดกายภาพ 10 ขั้นตอน
breathing_state = "IDLE" 
current_exercise_type = "neck"
current_step_idx = 0
current_phase = 1 # 1: Inhale, 2: Hold, 3: Exhale
time_left = 3
instruction_en = "Get Ready"
target_breathing = "INHALE"

# ตัวแปรสำหรับระบบเวลาอัจฉริยะ (อิงจากเฟรมภาพ)
last_frame_time = 0.0
elapsed_phase = 0.0
total_session_time = 120.0
is_session_complete = False

# ลำดับท่าทาง 10 ขั้นตอน
NECK_STEPS = [
    {"step": 1, "inst": "Clasp hands at the solar plexus", "type": "clasp"},
    {"step": 2, "inst": "Extend arms to the LEFT", "type": "left"},
    {"step": 3, "inst": "Clasp hands at the solar plexus", "type": "clasp"},
    {"step": 4, "inst": "Extend arms to the RIGHT", "type": "right"},
    {"step": 5, "inst": "Clasp hands at the solar plexus", "type": "clasp"},
    {"step": 6, "inst": "Extend arms FORWARD", "type": "forward"},
    {"step": 7, "inst": "Clasp hands at the solar plexus", "type": "clasp"},
    {"step": 8, "inst": "Extend arms UPWARD", "type": "upward"},
    {"step": 9, "inst": "Place hands on the CROWN", "type": "crown"},
    {"step": 10, "inst": "Clasp hands at the solar plexus", "type": "clasp"},
]

BACK_STEPS = [
    {"step": 1, "inst": "Twist torso and look back", "type": "back"},
    {"step": 2, "inst": "Release and breathe", "type": "rest"},
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def draw_target_hologram(frame, pose_type, cx, cy, scale, color):
    pts = {}
    if pose_type == 'clasp':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.8, 0.6), 'R_EL': (0.8, 0.6), 'L_WR': (-0.05, 0.6), 'R_WR': (0.05, 0.6)}
    elif pose_type == 'left':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-1.3, 0.1), 'R_EL': (0, 0.5), 'L_WR': (-1.8, 0), 'R_WR': (-1.8, 0)}
    elif pose_type == 'right':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (0, 0.5), 'R_EL': (1.3, 0.1), 'L_WR': (1.8, 0), 'R_WR': (1.8, 0)}
    elif pose_type == 'forward':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.8, 0.4), 'R_EL': (0.8, 0.4), 'L_WR': (0, 0.2), 'R_WR': (0, 0.2)}
    elif pose_type == 'upward':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.6, -1.0), 'R_EL': (0.6, -1.0), 'L_WR': (0, -1.8), 'R_WR': (0, -1.8)}
    elif pose_type == 'crown':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-1.1, -0.4), 'R_EL': (1.1, -0.4), 'L_WR': (0, -0.7), 'R_WR': (0, -0.7)}
    elif pose_type == 'back':
        pts = {'L_SH': (-0.3, 0), 'R_SH': (0.3, 0), 'L_EL': (-0.5, 0.8), 'R_EL': (0.8, 0.5), 'L_WR': (-0.4, 1.5), 'R_WR': (1.2, 0.8)}
    elif pose_type == 'rest':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.6, 0.8), 'R_EL': (0.6, 0.8), 'L_WR': (0, 0.6), 'R_WR': (0, 0.6)}
    
    if not pts: return
    pts['HEAD'], pts['SPINE'] = (0, -0.7), (0, 1.2)
    
    px_pts = {k: (int(cx + v[0] * scale), int(cy + v[1] * scale)) for k, v in pts.items()}
    
    for start, end in [('L_SH', 'L_EL'), ('L_EL', 'L_WR'), ('R_SH', 'R_EL'), ('R_EL', 'R_WR'), ('L_SH', 'R_SH')]:
        cv2.line(frame, px_pts[start], px_pts[end], color, 8)
    
    cv2.circle(frame, px_pts['HEAD'], int(scale * 0.4), color, 8)
    
    for k, v in px_pts.items():
        if k != 'HEAD':
            cv2.circle(frame, v, 10, (255, 255, 255), -1)
            cv2.circle(frame, v, 10, color, 3)

def tracking_loop():
    global tracking_active, output_frame, is_calibrating, calibration_start_time, baseline_shoulder_width, calibration_data_x
    global breathing_state, current_exercise_type, current_step_idx, current_phase, time_left, instruction_en, target_breathing, current_calib_msg, current_pose_msg
    global fhp_start_time, rounded_start_time, static_start_time, static_anchor
    global last_frame_time, elapsed_phase, total_session_time, is_session_complete
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while tracking_active:
            ret, frame = cap.read()
            current_time = time.time()
            
            if last_frame_time == 0: 
                last_frame_time = current_time
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            if dt > 1.0:
                dt = 0
                
            if not ret: 
                continue
                
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w, _ = frame.shape
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            if not results.pose_landmarks:
                current_calib_msg = "Please step into the frame"
                if breathing_state == "ACTIVE":
                    target_breathing = "PAUSED"
                    current_pose_msg = "STAY IN FRAME TO RESUME"
            else:
                lm = results.pose_landmarks.landmark
                ls, rs = lm[11], lm[12]
                le, re = lm[13], lm[14]
                lw, rw = lm[15], lm[16]
                nose, left_ear, right_ear = lm[0], lm[7], lm[8]
                curr_width = abs(ls.x - rs.x)
                
                core_visibility = min(nose.visibility, ls.visibility, rs.visibility)

                if brightness < 40:
                    current_calib_msg = "Low light detected"
                elif curr_width > 0.45:
                    current_calib_msg = "Please move back slightly"
                elif curr_width < 0.15:
                    current_calib_msg = "Please move closer"
                else:
                    if is_calibrating:
                        elapsed = current_time - calibration_start_time
                        if elapsed < calibration_duration:
                            calibration_data_x.append(curr_width)
                            current_calib_msg = f"Calibrating... {int(calibration_duration - elapsed) + 1}s"
                        else:
                            if len(calibration_data_x) > 0:
                                baseline_shoulder_width = sum(calibration_data_x) / len(calibration_data_x)
                            is_calibrating = False
                            calibration_data_x = []
                            current_calib_msg = "Calibration Complete"
                    else:
                        current_calib_msg = "All set!"

                for i in range(11): lm[i].visibility = 0
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, BODY_CONNECTIONS)

                if baseline_shoulder_width and not is_calibrating:
                    if breathing_state == "IDLE":
                        active_warnings = []
                        ear_y = (left_ear.y + right_ear.y) / 2.0
                        ear_z = (left_ear.z + right_ear.z) / 2.0
                        shoulder_y = (ls.y + rs.y) / 2.0
                        shoulder_z = (ls.z + rs.z) / 2.0
                        shoulder_x = (ls.x + rs.x) / 2.0

                        delta_y = shoulder_y - ear_y 
                        delta_z = shoulder_z - ear_z 
                        cva_angle = math.degrees(math.atan2(delta_y, delta_z)) if delta_z > 0 else 90.0

                        if cva_angle < 65.0:
                            if fhp_start_time is None: fhp_start_time = current_time
                            elif (current_time - fhp_start_time) >= FHP_TIME_LIMIT: active_warnings.append("Forward Head")
                        else: fhp_start_time = None 

                        if curr_width < (baseline_shoulder_width * 0.95):
                            if rounded_start_time is None: rounded_start_time = current_time
                            elif (current_time - rounded_start_time) >= ROUNDED_TIME_LIMIT: active_warnings.append("Rounded Shoulders")
                        else: rounded_start_time = None

                        current_pose_msg = " | ".join(active_warnings) if active_warnings else ""
                    else:
                        seq = NECK_STEPS if current_exercise_type == "neck" else BACK_STEPS
                        
                        if current_step_idx >= len(seq):
                            current_step_idx = 0 
                            
                        step_info = seq[current_step_idx]
                        instruction_en = step_info["inst"]
                        pt = step_info["type"]
                        
                        if core_visibility < 0.5:
                            target_breathing = "PAUSED"
                            current_pose_msg = "STAY IN FRAME TO RESUME"
                        else:
                            total_session_time -= dt
                            elapsed_phase += dt
                            
                            # [แก้ไขที่ 1] ปิดสถานะกายภาพทันทีเมื่อครบ 2 นาที เพื่อไม่ให้ค้างไปถึงหน้า Calibrate
                            if total_session_time <= 0:
                                is_session_complete = True
                                breathing_state = "IDLE"
                            
                            if current_phase == 1:
                                target_breathing = "INHALE"
                                phase_duration = 3
                                if elapsed_phase >= phase_duration:
                                    current_phase = 2
                                    elapsed_phase = 0
                            elif current_phase == 2:
                                target_breathing = "HOLD"
                                phase_duration = 5
                                if elapsed_phase >= phase_duration:
                                    current_phase = 3
                                    elapsed_phase = 0
                            elif current_phase == 3:
                                target_breathing = "EXHALE"
                                phase_duration = 3
                                if elapsed_phase >= phase_duration:
                                    current_phase = 1
                                    current_step_idx += 1
                                    elapsed_phase = 0

                            time_left = max(0, int(phase_duration - elapsed_phase))

                            is_perfect = False
                            wrist_dist = abs(lw.x - rw.x)
                            hands_clasped = wrist_dist < 0.12
                            la = calculate_angle([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                            ra = calculate_angle([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])

                            if pt == "clasp":
                                if hands_clasped and lw.y > ls.y - 0.1: is_perfect = True
                            elif pt == "left":
                                if hands_clasped and lw.x < ls.x - 0.15: is_perfect = True
                            elif pt == "right":
                                if hands_clasped and rw.x > rs.x + 0.15: is_perfect = True
                            elif pt == "forward":
                                if hands_clasped and la > 130 and ra > 130: is_perfect = True
                            elif pt == "upward":
                                if hands_clasped and lw.y < nose.y and la > 130: is_perfect = True
                            elif pt == "crown":
                                if hands_clasped and lw.y < ls.y and la < 110: is_perfect = True
                            elif pt == "back":
                                dx = abs(rs.x - ls.x)
                                dz = abs(rs.z - ls.z)
                                twist_angle = math.degrees(math.atan2(dz, dx + 1e-6))
                                chest_y = (ls.y + rs.y) / 2.0
                                if twist_angle >= 12 and ((lw.y > chest_y) or (rw.y > chest_y)): is_perfect = True
                            elif pt == "rest":
                                is_perfect = True 

                            cx, cy = int((ls.x + rs.x)/2 * w), int((ls.y + rs.y)/2 * h)
                            scale = baseline_shoulder_width * w
                            color = (0, 255, 0) if is_perfect else (0, 165, 255)
                            draw_target_hologram(frame, pt, cx, cy, scale, color)

                            current_pose_msg = "PERFECT!" if is_perfect else "ADJUST POSE"

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret: output_frame = buffer.tobytes()
            time.sleep(0.01)
    finally: cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
            time.sleep(0.016)
    finally:
        if started_by_feed and not manual_session:
            tracking_active = False

@app.route('/api/status')
def get_status():
    global target_breathing
    return jsonify({
        "calib_msg": current_calib_msg, 
        "pose_msg": current_pose_msg, 
        "instruction": instruction_en, 
        "breathing": target_breathing if breathing_state == "ACTIVE" else "IDLE", 
        "time_left": time_left,
        "total_time": max(0, int(total_session_time)), 
        "is_complete": is_session_complete
    })

@app.route('/api/start_pose', methods=['POST'])
def start_pose():
    global breathing_state, current_exercise_type, current_step_idx, current_phase
    global elapsed_phase, total_session_time, last_frame_time, is_session_complete
    data = request.json or {}
    current_exercise_type = data.get('type', 'neck')
    breathing_state = "ACTIVE"
    current_step_idx = 0
    current_phase = 1
    
    elapsed_phase = 0.0
    total_session_time = 120.0
    last_frame_time = time.time()
    is_session_complete = False
    return jsonify({"status": "success"})

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
    global breathing_state 
    
    # [แก้ไขที่ 2] บังคับปิดโหมดกายภาพทันทีที่กด Calibrate เพื่อให้ Skeleton หายไป
    breathing_state = "IDLE"

    if not is_calibrating:
        is_calibrating = True
        calibration_start_time = time.time()
        calibration_data_x.clear()
        baseline_shoulder_width = None
        return jsonify({"status": "success"})
    return jsonify({"status": "ignored"})

if __name__ == '__main__':
    print(">> Backend Server Running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)