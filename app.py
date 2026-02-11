import cv2
import threading
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from models.detection_model import CrowdDetector
from models.emotion_detection import EmotionDetector
from models.behavior_model import BehaviorRecognizer
from utils.heatmap_generator import generate_heatmap
from utils.alert_system import check_alert

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize models
detector = CrowdDetector(model_path='yolov8n.pt')
emotion_detector = EmotionDetector()
behavior_recognizer = BehaviorRecognizer()

cap = cv2.VideoCapture(0)  # Change to video file or camera feed

# Function to process video frame and emit results
def process_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes = detector.detect(frame)
        person_count = sum(1 for cls in classes if int(cls) == 0)

        frame_with_heatmap = generate_heatmap(frame, boxes)

        # Loop over the detected boxes and check emotion for faces
        for box in boxes:
            x1, y1, x2, y2 = box
            face_image = frame[int(y1):int(y2), int(x1):int(x2)]
            emotion = emotion_detector.detect(face_image)  # Detect emotion
            cv2.putText(frame_with_heatmap, f'Emotion: {emotion}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Detect behavior
        behavior = behavior_recognizer.recognize(frame)
        cv2.putText(frame_with_heatmap, f'Behavior: {behavior}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to send to frontend
        ret, jpeg = cv2.imencode('.jpg', frame_with_heatmap)
        if ret:
            frame_bytes = jpeg.tobytes()
            socketio.emit('video_feed', {'data': frame_bytes})  # Send the frame via WebSocket
        time.sleep(0.1)  # Adjust frame rate

# Route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Frontend page to display video

# Start processing frames in a separate thread
thread = threading.Thread(target=process_frame)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
