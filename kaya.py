from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

BODY_CONNECTIONS = frozenset([conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] > 10 and conn[1] > 10])

# === ตัวแปรระบบ Login (Mock Database) ===
users_db = {}

# Variables พื้นฐานจาก Local
baseline_shoulder_width = None
is_calibrating = False
calibration_start_time = 0
calibration_duration = 3
calibration_data_x = []
current_calib_msg = "Waiting for camera..."
current_pose_msg = ""
tracking_active = False
app_mode = "IDLE"

# โหมดออฟฟิศซินโดรม
FHP_TIME_LIMIT = 2.0       
ROUNDED_TIME_LIMIT = 2.0   
STATIC_TIME_LIMIT = 5.0   
fhp_start_time = None
rounded_start_time = None
static_start_time = None

# โหมดกายภาพ
breathing_state = "IDLE" 
current_exercise_type = "neck"
current_step_idx = 0
current_phase = 1 
time_left = 3
instruction_en = "Get Ready"
target_breathing = "INHALE"

# ตัวแปรเวลาและคะแนน
last_frame_time = 0.0
elapsed_phase = 0.0
total_session_time = 120.0
is_session_complete = False
daily_score = 100.0
active_time_sec = 0.0
stretch_count = 0
snooze_count = 0

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
    {"step": 1, "inst": "Left hand on waist, turn head fully LEFT", "type": "back_left"},
    {"step": 2, "inst": "Right hand on waist, turn head fully RIGHT", "type": "back_right"},
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def draw_target_hologram(frame, pose_type, cx, cy, scale, color):
    pts = {}
    if pose_type == 'clasp':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.6, 0.4), 'R_EL': (0.6, 0.4), 'L_WR': (-0.05, 0.4), 'R_WR': (0.05, 0.4)}
    elif pose_type == 'left':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-1.1, 0.1), 'R_EL': (0, 0.5), 'L_WR': (-1.4, 0), 'R_WR': (-1.4, 0)}
    elif pose_type == 'right':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (0, 0.5), 'R_EL': (1.1, 0.1), 'L_WR': (1.4, 0), 'R_WR': (1.4, 0)}
    elif pose_type == 'back_left':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.8, 0.5), 'R_EL': (0.6, 0.8), 'L_WR': (-0.6, 0.9), 'R_WR': (0.6, 1.2)}
        pts['HEAD'] = (-0.2, -0.7)
    elif pose_type == 'back_right':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.6, 0.8), 'R_EL': (0.8, 0.5), 'L_WR': (-0.6, 1.2), 'R_WR': (0.6, 0.9)}
        pts['HEAD'] = (0.2, -0.7)
    elif pose_type == 'ready':
        pts = {'L_SH': (-0.5, 0), 'R_SH': (0.5, 0), 'L_EL': (-0.5, 0.5), 'R_EL': (0.5, 0.5), 'L_WR': (-0.5, 1.0), 'R_WR': (0.5, 1.0)}
    
    if not pts: return
    if 'HEAD' not in pts: pts['HEAD'] = (0, -0.7)
    pts['SPINE'] = (0, 1.2)
    px_pts = {k: (int(cx + v[0] * scale), int(cy + v[1] * scale)) for k, v in pts.items()}
    for start, end in [('L_SH', 'L_EL'), ('L_EL', 'L_WR'), ('R_SH', 'R_EL'), ('R_EL', 'R_WR'), ('L_SH', 'R_SH')]:
        cv2.line(frame, px_pts[start], px_pts[end], color, 8)
    cv2.circle(frame, px_pts['HEAD'], int(scale * 0.4), color, 8)
    for k, v in px_pts.items():
        if k != 'HEAD':
            cv2.circle(frame, v, 10, (255, 255, 255), -1)
            cv2.circle(frame, v, 10, color, 3)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global tracking_active, is_calibrating, calibration_start_time, baseline_shoulder_width, calibration_data_x
    global breathing_state, current_exercise_type, current_step_idx, current_phase, time_left, instruction_en, target_breathing, current_calib_msg, current_pose_msg
    global fhp_start_time, rounded_start_time, static_start_time
    global last_frame_time, elapsed_phase, total_session_time, is_session_complete, app_mode
    global daily_score, active_time_sec, stretch_count

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400

    img_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return jsonify({"error": "Invalid image"}), 400

    current_time = time.time()
    if last_frame_time == 0: last_frame_time = current_time
    dt = current_time - last_frame_time
    last_frame_time = current_time
    if dt > 1.0: dt = 0 

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
        lh, rh = lm[23], lm[24]
        nose = lm[0]
        
        curr_width = abs(ls.x - rs.x)
        core_visibility = min(nose.visibility, ls.visibility, rs.visibility)

        if brightness < 40: current_calib_msg = "Low light detected"
        elif curr_width > 0.45: current_calib_msg = "Please move back slightly"
        elif curr_width < 0.15: current_calib_msg = "Please move closer"
        else:
            if is_calibrating:
                elapsed = current_time - calibration_start_time
                if elapsed < calibration_duration:
                    calibration_data_x.append(curr_width)
                    current_calib_msg = f"Calibrating... {int(calibration_duration - elapsed) + 1}s"
                else:
                    if len(calibration_data_x) > 0: baseline_shoulder_width = sum(calibration_data_x) / len(calibration_data_x)
                    is_calibrating = False
                    app_mode = "IDLE"
                    current_calib_msg = "Calibration Complete"
            else: current_calib_msg = "All set!"

        for i in range(11): lm[i].visibility = 0
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, BODY_CONNECTIONS)

        if baseline_shoulder_width and not is_calibrating:
            if breathing_state == "IDLE":
                if tracking_active:
                    active_time_sec += dt
                    # ใช้ Logic ฉบับ Local ดั้งเดิมของคุณ
                    if curr_width < (baseline_shoulder_width * 0.95):
                        if rounded_start_time is None: rounded_start_time = current_time
                        elif (current_time - rounded_start_time) >= ROUNDED_TIME_LIMIT: current_pose_msg = "Rounded Shoulders"
                    else: rounded_start_time = None

            elif app_mode == "SESSION":
                seq = NECK_STEPS if current_exercise_type == "neck" else BACK_STEPS
                if current_step_idx >= len(seq): current_step_idx = 0 
                step_info = seq[current_step_idx]
                
                instruction_en = step_info["inst"]
                pt = step_info["type"]
                
                if core_visibility < 0.5:
                    target_breathing = "PAUSED"
                    current_pose_msg = "STAY IN FRAME TO RESUME"
                else:
                    total_session_time -= dt
                    elapsed_phase += dt
                    if total_session_time <= 0:
                        is_session_complete = True
                        breathing_state = "IDLE"
                        stretch_count += 1
                    
                    if current_phase == 1:
                        target_breathing = "INHALE"
                        if elapsed_phase >= 3: current_phase, elapsed_phase = 2, 0
                    elif current_phase == 2:
                        target_breathing = "HOLD"
                        if elapsed_phase >= 7: current_phase, elapsed_phase = 3, 0 
                    elif current_phase == 3:
                        target_breathing = "EXHALE"
                        if current_exercise_type != "neck":
                            pt = "ready"
                            instruction_en = "Return to READY pose"
                        if elapsed_phase >= 3: current_phase, elapsed_phase, current_step_idx = 1, 0, current_step_idx + 1

                    time_limit = 7 if current_phase == 2 else 3
                    time_left = max(0, int(time_limit - elapsed_phase))

                    is_perfect = False
                    if pt == "clasp":
                        if abs(lw.x - rw.x) < 0.12 and lw.y > ls.y - 0.1: is_perfect = True
                    elif pt == "back_left":
                        if lw.y > ls.y and lw.y < lh.y and nose.x < (ls.x + rs.x)/2 - 0.05: is_perfect = True
                    elif pt == "back_right":
                        if rw.y > rs.y and rw.y < rh.y and nose.x > (ls.x + rs.x)/2 + 0.05: is_perfect = True
                    elif pt == "ready":
                        if lw.y > ls.y and rw.y > rs.y and abs(lw.x - ls.x) < 0.15: is_perfect = True

                    cx, cy = int((ls.x + rs.x)/2 * w), int((ls.y + rs.y)/2 * h)
                    draw_target_hologram(frame, pt, cx, cy, baseline_shoulder_width * w, (0, 255, 0) if is_perfect else (0, 165, 255))
                    current_pose_msg = "PERFECT!" if is_perfect else "ADJUST POSE"

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": out_b64, "calib_msg": current_calib_msg, "pose_msg": current_pose_msg,
        "instruction": instruction_en, "breathing": target_breathing if breathing_state == "ACTIVE" else "IDLE", 
        "time_left": time_left, "total_time": max(0, int(total_session_time)), "is_complete": is_session_complete,
        "daily_score": int(daily_score), "active_time_sec": active_time_sec, "stretch_count": stretch_count, "snooze_count": snooze_count
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    user = data.get('username')
    pwd = data.get('password')
    if not user or not pwd: return jsonify({"status": "error", "message": "Missing info"}), 400
    if user in users_db: return jsonify({"status": "error", "message": "Username already exists"}), 400
    users_db[user] = pwd
    return jsonify({"status": "success"})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = data.get('username')
    pwd = data.get('password')
    if users_db.get(user) == pwd: return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid username or password"}), 401

@app.route('/api/status')
def get_status():
    return jsonify({
        "calib_msg": current_calib_msg, "pose_msg": current_pose_msg, "instruction": instruction_en, 
        "breathing": target_breathing if breathing_state == "ACTIVE" else "IDLE", 
        "time_left": time_left, "total_time": max(0, int(total_session_time)), "is_complete": is_session_complete,
        "daily_score": int(daily_score), "active_time_sec": active_time_sec, "stretch_count": stretch_count, "snooze_count": snooze_count
    })

@app.route('/api/start_pose', methods=['POST'])
def start_pose():
    global breathing_state, current_exercise_type, current_step_idx, current_phase, app_mode
    global elapsed_phase, total_session_time, last_frame_time, is_session_complete
    data = request.json or {}
    current_exercise_type = data.get('type', 'neck')
    app_mode, breathing_state, current_step_idx, current_phase, elapsed_phase, total_session_time, last_frame_time, is_session_complete = "SESSION", "ACTIVE", 0, 1, 0.0, 120.0, time.time(), False
    return jsonify({"status": "success"})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global is_calibrating, calibration_start_time, calibration_data_x, baseline_shoulder_width, breathing_state, app_mode
    breathing_state, app_mode = "IDLE", "CALIBRATE"
    if not is_calibrating:
        is_calibrating, calibration_start_time = True, time.time()
        calibration_data_x.clear()
        baseline_shoulder_width = None
        return jsonify({"status": "success"})
    return jsonify({"status": "ignored"})

@app.route('/api/toggle_session', methods=['POST'])
def toggle_session():
    global tracking_active, app_mode, daily_score, active_time_sec, stretch_count, snooze_count 
    data = request.json or {}
    action = data.get('action')
    if action == 'start':
        if 'init_score' in data: daily_score = float(data['init_score'])
        if 'init_active_time' in data: active_time_sec = float(data['init_active_time'])
        if 'init_stretch' in data: stretch_count = int(data['init_stretch'])
        if 'init_snooze' in data: snooze_count = int(data['init_snooze'])
        tracking_active, app_mode = True, "MONITORING" 
        return jsonify({"status": "started", "active": True})
    elif action == 'stop':
        tracking_active, app_mode = False, "IDLE"
        return jsonify({"status": "stopped", "active": False})
    return jsonify({"status": "invalid action"}), 400

@app.route('/api/session_status', methods=['GET'])
def session_status():
    global tracking_active
    return jsonify({"active": tracking_active})

@app.route('/api/snooze', methods=['POST'])
def register_snooze():
    global snooze_count, daily_score
    snooze_count += 1
    daily_score = max(0.0, daily_score - 1.0)
    return jsonify({"status": "snoozed", "score": daily_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)