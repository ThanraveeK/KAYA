from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

users_db = {}
daily_score = 100.0
active_time_sec = 0.0
stretch_count = 0
snooze_count = 0
tracking_active = False

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
        "daily_score": int(daily_score),
        "active_time_sec": active_time_sec,
        "stretch_count": stretch_count,
        "snooze_count": snooze_count,
        "pose_msg": "" # Backend ไม่ประมวลผลภาพแล้ว
    })

@app.route('/api/toggle_session', methods=['POST'])
def toggle_session():
    global tracking_active, daily_score, active_time_sec, stretch_count, snooze_count
    data = request.json or {}
    action = data.get('action')
    if action == 'start':
        if 'init_score' in data: daily_score = float(data['init_score'])
        if 'init_active_time' in data: active_time_sec = float(data['init_active_time'])
        if 'init_stretch' in data: stretch_count = int(data['init_stretch'])
        if 'init_snooze' in data: snooze_count = int(data['init_snooze'])
        tracking_active = True
        return jsonify({"status": "started", "active": True})
    elif action == 'stop':
        tracking_active = False
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

# ปล่อย Endpoint เหล่านี้ไว้เป็น Dummy ดักไว้เฉยๆ ไม่ให้ Error
@app.route('/api/start_pose', methods=['POST'])
def start_pose(): return jsonify({"status": "success"})
@app.route('/api/calibrate', methods=['POST'])
def calibrate(): return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)