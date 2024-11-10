from cog import BasePredictor, Input, Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort import DeepSort
import tempfile

class Predictor(BasePredictor):
    def setup(self):
        """Load models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load YOLO models
        self.person_model = YOLO("models/person_model.pt").to(self.device)
        self.ball_model = YOLO("models/ball_model.pt").to(self.device)
        
        # Initialize DeepSORT
        self.player_tracker = DeepSort(
            model_path="deep_sort/deep/checkpoint/ckpt.t7",
            max_dist=0.2,
            min_confidence=0.3,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
            use_cuda=(self.device.type == "cuda")
        )

    def predict(
        self,
        mode: str = Input(
            description="Choose mode: 'live' for camera or 'upload' for video file",
            choices=["live", "upload"],
            default="upload"
        ),
        video: Path = Input(
            description="Input video file (only for upload mode)",
            default=None
        ),
        stream_url: str = Input(
            description="Stream URL (for live mode, e.g., RTSP stream)",
            default=None
        ),
        confidence_threshold: float = Input(
            description="Detection confidence threshold",
            default=0.3,
            ge=0.0,
            le=1.0,
        )
    ) -> Path:
        """Process video in either live or upload mode"""
        # Initialize video capture based on mode
        if mode == "live":
            if stream_url:
                cap = cv2.VideoCapture(stream_url)
            else:
                cap = cv2.VideoCapture(0)  # Default camera
            print("Starting live camera feed...")
        else:
            if not video:
                raise ValueError("Video file required for upload mode")
            cap = cv2.VideoCapture(str(video))
            print(f"Processing video file: {video}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        unique_player_ids = set()
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detections
                person_results = self.person_model(frame, conf=confidence_threshold)
                ball_results = self.ball_model(frame, conf=confidence_threshold)

                # Process person detections
                person_boxes = []
                person_confidences = []

                for result in person_results[0].boxes:
                    try:
                        cls_id = int(result.cls[0])
                        class_name = self.person_model.names[cls_id].lower()
                        box = result.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        conf = float(result.conf[0])

                        if class_name == 'person':
                            w = x2 - x1
                            h = y2 - y1
                            person_boxes.append([x1 + w/2, y1 + h/2, w, h])
                            person_confidences.append(conf)
                    except Exception as e:
                        print(f"Error processing person: {e}")

                # Process ball detections
                for result in ball_results[0].boxes:
                    try:
                        box = result.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Error processing ball: {e}")

                # Update tracking
                if person_boxes:
                    person_boxes = np.array(person_boxes)
                    person_confidences = np.array(person_confidences)
                    tracks = self.player_tracker.update(person_boxes, person_confidences, frame)

                    for track in tracks:
                        x1, y1, x2, y2, track_id = track.astype(int)
                        unique_player_ids.add(track_id)

                        color = (int((track_id * 47) % 255),
                                int((track_id * 89) % 255),
                                int((track_id * 137) % 255))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Player-{track_id}",
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, color, 2)

                # Add statistics
                cv2.putText(frame, f"Players: {len(unique_player_ids)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)

                # Write frame
                out.write(frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")

                # For live mode, return after a certain duration or number of frames
                if mode == "live" and frame_count >= 300:  # 10 seconds at 30 fps
                    break

        finally:
            cap.release()
            out.release()
            print("Processing complete!")

        return output_path