# app.py
from flask import Flask, render_template, Response, request, send_file, jsonify
from detector.realtime_detector import RealtimeDetector
import os
import tempfile
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize detector with model paths
detector = RealtimeDetector(
    person_model_path=os.getenv('PERSON_MODEL_PATH', 'models/person_model.pt'),
    ball_model_path=os.getenv('BALL_MODEL_PATH', 'models/ball_model.pt'),
    deep_sort_weights=os.getenv('DEEPSORT_WEIGHTS', 'deep_sort/deep/checkpoint/ckpt.t7')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for live video feed"""
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_tracking')
def stop_tracking():
    """Stop live video tracking"""
    detector.stop_streaming()
    return jsonify({'status': 'stopped'})

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process uploaded video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Process video
        output_path = detector.process_video(filepath)
        
        # Return processed video
        return send_file(output_path, 
                        as_attachment=True,
                        download_name='processed_video.mp4')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup temporary files
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)