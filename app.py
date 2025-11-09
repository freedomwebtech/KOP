from flask import Flask, render_template, Response, jsonify, request
import cv2
import random
import json
import os
import csv
from datetime import datetime
from shapely.geometry import LineString, Point
from ultralytics import YOLO
from collections import defaultdict
import threading
import numpy as np
import imutils
from imutils.video import VideoStream

app = Flask(__name__)

# ------------------- Global Variables -------------------
model = YOLO("bestkop.pt")
names = model.names

vs = None  # VideoStream object for RTSP
dots = {}
track_history = defaultdict(list)
dot_counter = 1
drawing_line = False
temp_line_points = []
video_running = False
lock = threading.Lock()

line_one = None
line_two = None
track_line_state = {}

LINE_CONFIG_FILE = "line_config.json"
CSV_FILE = "truck_counts.csv"
RTSP_CONFIG_FILE = "rtsp_config.txt"

# ------------------- CSV Functions -------------------
def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Number Plate', 'Mode', 'IN Count', 'OUT Count', 'Total Count'])
        print(f"✅ CSV file created: {CSV_FILE}")

def save_to_csv(numberplate, mode, in_count, out_count):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total = in_count + out_count
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, numberplate, mode, in_count, out_count, total])
        print(f"✅ Saved: {numberplate} | Mode: {mode} | IN: {in_count} | OUT: {out_count}")
        return True
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
        return False

# ------------------- Line Storage Functions -------------------
def load_lines_from_file():
    global line_one, line_two
    try:
        if os.path.exists(LINE_CONFIG_FILE):
            with open(LINE_CONFIG_FILE, 'r') as f:
                data = json.load(f)
                if data.get('line_one'):
                    line_one = [tuple(p) for p in data['line_one']]
                if data.get('line_two'):
                    line_two = [tuple(p) for p in data['line_two']]
                print(f"✅ Lines loaded - Line 1: {line_one}, Line 2: {line_two}")
    except Exception as e:
        print(f"❌ Error loading lines: {e}")

def save_lines_to_file():
    try:
        with open(LINE_CONFIG_FILE, 'w') as f:
            json.dump({'line_one': line_one, 'line_two': line_two}, f)
        print(f"✅ Lines saved")
    except Exception as e:
        print(f"❌ Error saving lines: {e}")

def delete_lines_from_file():
    try:
        if os.path.exists(LINE_CONFIG_FILE):
            os.remove(LINE_CONFIG_FILE)
            print(f"✅ Lines deleted")
    except Exception as e:
        print(f"❌ Error deleting lines: {e}")

# ------------------- Line Crossing Detection -------------------
def check_line_crossing(prev_pos, curr_pos, track_id):
    global line_one, line_two, track_line_state
    
    if not line_one or not line_two or len(line_one) != 2 or len(line_two) != 2:
        return
    
    movement_line = LineString([prev_pos, curr_pos])
    line1_geometry = LineString(line_one)
    line2_geometry = LineString(line_two)
    
    crosses_line1 = movement_line.intersects(line1_geometry)
    crosses_line2 = movement_line.intersects(line2_geometry)
    
    with lock:
        if crosses_line1:
            if track_id not in track_line_state:
                track_line_state[track_id] = 'line1'
            elif track_line_state[track_id] == 'line2':
                for dot_id, data in dots.items():
                    if data.get("mode") == "OUT":
                        data["OUT"] += 1
                del track_line_state[track_id]
        
        if crosses_line2:
            if track_id not in track_line_state:
                track_line_state[track_id] = 'line2'
            elif track_line_state[track_id] == 'line1':
                for dot_id, data in dots.items():
                    if data.get("mode") == "IN":
                        data["IN"] += 1
                del track_line_state[track_id]

# ------------------- Frame Generation -------------------
def generate_frames():
    global vs, video_running, line_one, line_two
    
    while video_running:
        if vs is None:
            break
            
        frame = vs.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1020, 600))

        results = model.track(frame, persist=True)
        detections = results[0]

        if detections.boxes.id is not None:
            ids = detections.boxes.id.cpu().numpy().astype(int)
            boxes = detections.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = detections.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, ids, class_ids):
                name = names[cls_id].lower()
                if "package" not in name:
                    continue

                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                if len(track_history[track_id]) > 1:
                    prev_pos = track_history[track_id][-2]
                    curr_pos = (cx, cy)
                    check_line_crossing(prev_pos, curr_pos, track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        with lock:
            for did, data in dots.items():
                color = data["color"]
                cx, cy = data["center"]
                mode = data.get("mode", "IN")
                
                cv2.circle(frame, (cx, cy), 8, color, -1)
                
                if mode == "IN":
                    cv2.putText(frame, f"{data.get('numberplate', '')} [IN] | IN:{data['IN']}",
                               (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.putText(frame, f"{data.get('numberplate', '')} [OUT] | OUT:{data['OUT']}",
                               (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if line_one and len(line_one) == 2:
                cv2.line(frame, line_one[0], line_one[1], (0, 255, 255), 4)
                cv2.putText(frame, "LINE 1", (line_one[0][0], line_one[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            
            if line_two and len(line_two) == 2:
                cv2.line(frame, line_two[0], line_two[1], (255, 0, 255), 4)
                cv2.putText(frame, "LINE 2", (line_two[0][0], line_two[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3)

            if drawing_line:
                cv2.putText(frame, "Drawing line - Click 2 points",
                           (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                for pt in temp_line_points:
                    cv2.circle(frame, pt, 5, (0, 255, 255), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global vs, video_running
    
    data = request.json
    rtsp_url = data.get('rtsp_url', '').strip()
    
    if not rtsp_url:
        return jsonify({'status': 'error', 'message': 'Please enter RTSP URL'})
    
    if not rtsp_url.startswith('rtsp://'):
        return jsonify({'status': 'error', 'message': 'URL must start with rtsp://'})
    
    if vs is not None:
        try:
            vs.stop()
        except:
            pass
        vs = None
    
    try:
        print(f"📡 Connecting to RTSP: {rtsp_url}")
        vs = VideoStream(rtsp_url).start()
        import time
        time.sleep(2.0)
        
        video_running = True
        print("✅ RTSP connected")
        return jsonify({'status': 'success', 'message': 'RTSP stream connected'})
        
    except Exception as e:
        print(f"❌ RTSP Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global vs, video_running
    
    video_running = False
    if vs is not None:
        try:
            vs.stop()
        except:
            pass
        vs = None
    
    return jsonify({'status': 'success', 'message': 'Stream stopped'})

@app.route('/create_dot', methods=['POST'])
def create_dot():
    global dot_counter
    data = request.json
    x, y = data['x'], data['y']
    numberplate = data.get('numberplate', '').strip().upper()
    mode = data.get('mode', 'IN')
    
    if not numberplate:
        return jsonify({'status': 'error', 'message': 'Number plate required'})
    
    with lock:
        for existing_id, existing_data in dots.items():
            if existing_data.get('numberplate') == numberplate:
                return jsonify({'status': 'error', 'message': f'{numberplate} already exists'})
        
        color = [random.randint(0, 255) for _ in range(3)]
        dots[dot_counter] = {
            "center": (x, y),
            "color": color,
            "numberplate": numberplate,
            "mode": mode,
            "IN": 0,
            "OUT": 0
        }
        dot_id = dot_counter
        dot_counter += 1
    
    return jsonify({'status': 'success', 'dot_id': dot_id, 'numberplate': numberplate, 'mode': mode})

@app.route('/finish_and_save', methods=['POST'])
def finish_and_save():
    data = request.json
    dot_id = data.get('dot_id')
    
    if not dot_id:
        return jsonify({'status': 'error', 'message': 'Dot ID required'})
    
    with lock:
        dot_id = int(dot_id)
        if dot_id not in dots:
            return jsonify({'status': 'error', 'message': 'Truck not found'})
        
        truck_data = dots[dot_id]
        numberplate = truck_data.get('numberplate', 'UNKNOWN')
        mode = truck_data.get('mode', 'IN')
        in_count = truck_data.get('IN', 0)
        out_count = truck_data.get('OUT', 0)
        
        if save_to_csv(numberplate, mode, in_count, out_count):
            del dots[dot_id]
            return jsonify({
                'status': 'success', 
                'message': f'{numberplate} saved to CSV',
                'data': {'numberplate': numberplate, 'mode': mode, 'in_count': in_count, 'out_count': out_count}
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save'})

@app.route('/remove_truck', methods=['POST'])
def remove_truck():
    data = request.json
    dot_id = data.get('dot_id')
    
    if not dot_id:
        return jsonify({'status': 'error', 'message': 'Dot ID required'})
    
    with lock:
        dot_id = int(dot_id)
        if dot_id not in dots:
            return jsonify({'status': 'error', 'message': 'Truck not found'})
        
        numberplate = dots[dot_id].get('numberplate', 'UNKNOWN')
        del dots[dot_id]
        return jsonify({'status': 'success', 'message': f'{numberplate} removed'})

@app.route('/add_line_point', methods=['POST'])
def add_line_point():
    global drawing_line, temp_line_points, line_one, line_two
    data = request.json
    x, y = data['x'], data['y']
    line_number = data.get('line_number', 1)
    
    with lock:
        if drawing_line:
            temp_line_points.append((x, y))
            if len(temp_line_points) == 2:
                if line_number == 1:
                    line_one = temp_line_points.copy()
                else:
                    line_two = temp_line_points.copy()
                
                if line_one and line_two:
                    save_lines_to_file()
                
                temp_line_points = []
                drawing_line = False
                return jsonify({'status': 'complete', 'line_number': line_number})
    
    return jsonify({'status': 'added', 'points': len(temp_line_points)})

@app.route('/start_line_drawing', methods=['POST'])
def start_line_drawing():
    global drawing_line, temp_line_points
    data = request.json
    line_number = data.get('line_number', 1)
    
    with lock:
        drawing_line = True
        temp_line_points = []
        return jsonify({'status': 'success', 'message': f'Click 2 points for Line {line_number}'})

@app.route('/delete_lines', methods=['POST'])
def delete_lines():
    global line_one, line_two
    with lock:
        line_one = None
        line_two = None
        delete_lines_from_file()
        return jsonify({'status': 'success', 'message': 'Lines deleted'})

@app.route('/reset', methods=['POST'])
def reset():
    global dots, track_history, dot_counter, drawing_line, temp_line_points, track_line_state
    with lock:
        dots.clear()
        track_history.clear()
        track_line_state.clear()
        dot_counter = 1
        drawing_line = False
        temp_line_points = []
    return jsonify({'status': 'success', 'message': 'Reset complete'})

@app.route('/get_counts', methods=['GET'])
def get_counts():
    with lock:
        counts = {dot_id: {
            'IN': data['IN'], 
            'OUT': data['OUT'], 
            'numberplate': data.get('numberplate', ''),
            'mode': data.get('mode', 'IN')
        } for dot_id, data in dots.items()}
    return jsonify(counts)

@app.route('/get_line_status', methods=['GET'])
def get_line_status():
    with lock:
        return jsonify({
            'line_one_drawn': line_one is not None,
            'line_two_drawn': line_two is not None
        })

if __name__ == '__main__':
    initialize_csv()
    load_lines_from_file()
    
    print("=" * 60)
    print("🚀 RTSP Live Stream Package Counter")
    print("=" * 60)
    print(f"📊 CSV: {CSV_FILE}")
    print(f"📏 Lines: {LINE_CONFIG_FILE}")
    print("🌐 Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)