from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import os
from ultralytics import YOLO
import pytesseract
import time
from threading import Thread

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app)

# Ensure necessary directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Specify the path to Tesseract-OCR
pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
model = YOLO('yolov8n.pt')

# Detection parameters
line_y_position = 300  # Position of detection line
line_color = (0, 255, 0)  # Green line
vehicle_classes = [1, 2, 3, 4, 6, 8]  # Classes for cars, trucks, motorcycles, etc.

def process_video_frame(frame):
    """
    Process a single video frame to detect vehicles and violations, and return
    the processed frame along with vehicle counts.
    """
    # Resize the frame for consistent display
    frame_resized = cv2.resize(frame, (640, 480))

    # Process the frame with YOLO model
    results = model(frame_resized)
    vehicle_count = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[cls_id]}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Count detected vehicles
            vehicle_count += 1

    # Add counts to the frame
    cv2.putText(frame_resized, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add signal and line
    cv2.line(frame_resized, (0, line_y_position), (640, line_y_position), line_color, 2)
    cv2.putText(frame_resized, "Signal: Green", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame_resized, vehicle_count

def stream_video_frames(video_path, video_id):
    """
    Process and stream video frames using WebSocket for multiple video streams.
    Each frame is processed, and vehicle counts are sent to the frontend.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and get vehicle counts
        processed_frame, vehicle_count = process_video_frame(frame)

        # Encode the processed frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Error: Could not encode frame.")
            break
        frame_bytes = jpeg.tobytes()

        # Send frame and vehicle counts over WebSocket for the given video_id
        socketio.emit('video_frame', {'frame': frame_bytes, 'vehicles': vehicle_count, 'video_id': video_id})

        time.sleep(0.1)  # Slow down the stream to avoid overwhelming the client

    cap.release()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/project')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start streaming."""
    videos = request.files.getlist('video')
    if len(videos) > 4:
        return 'You can only upload up to 4 videos.', 400

    video_paths = []
    for video in videos:
        video_path = os.path.join('uploads', video.filename)
        video.save(video_path)
        video_paths.append(video_path)

    # Start video processing and streaming via WebSocket for each video
    for index, video_path in enumerate(video_paths):
        socketio.start_background_task(target=stream_video_frames, video_path=video_path, video_id=index)

    return render_template('index.html', video_paths=video_paths)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
