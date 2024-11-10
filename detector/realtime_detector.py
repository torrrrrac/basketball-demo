import torch
import cv2
from ultralytics import YOLO
import time
import numpy as np
from deep_sort import DeepSort
import threading
from queue import Queue
import base64

class RealtimeDetector:
    def __init__(self, person_model_path, ball_model_path, deep_sort_weights, target_width=854, target_height=480):
        
        print(f"Loading person model from: {person_model_path}")
        self.person_model = YOLO(person_model_path)
        
        print(f"Loading basketball model from: {ball_model_path}")
        self.ball_model = YOLO(ball_model_path)
        
        
        self.player_tracker = DeepSort(
            model_path=deep_sort_weights,
            max_dist=0.2,
            min_confidence=0.6,
            max_iou_distance=0.7,
            max_age=15,
            n_init=3,
            nn_budget=100
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.target_width = target_width
        self.target_height = target_height
        
        
        self.unique_player_ids = set()
        self.fps_start_time = time.time()
        self.fps_counter = 0

        
        self.is_streaming = False
        self.frame_queue = Queue(maxsize=10)
        self.stream_thread = None

        
        self.class_colors = {
            'person': (0, 255, 0),    
            'ball': (0, 0, 255),      
            'rim': (255, 0, 0),       
            'shoot': (255, 255, 0)    
        }

    def process_frame(self, frame):
        try:
            
            person_results = self.person_model(frame, conf=0.5)
            ball_results = self.ball_model(frame, conf=0.35)
            
            
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
                    else:
                        color = self.class_colors.get(class_name, (128, 128, 128))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
                        
                except Exception as e:
                    print(f"Error processing person detection: {e}")
            
            
            for result in ball_results[0].boxes:
                try:
                    box = result.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(result.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"Ball {conf:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error processing ball detection: {e}")
            
            
            if person_boxes:
                person_boxes = np.array(person_boxes)
                person_confidences = np.array(person_confidences)
                person_tracks = self.player_tracker.update(person_boxes, person_confidences, frame)
                
                for track in person_tracks:
                    x1, y1, x2, y2, track_id = track.astype(int)
                    self.unique_player_ids.add(track_id)
                    color = (int((track_id * 47) % 255), 
                            int((track_id * 89) % 255), 
                            int((track_id * 137) % 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Player-{track_id}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
            
            
            cv2.putText(frame, f"Players: {len(self.unique_player_ids)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.fps_counter += 1
            if (time.time() - self.fps_start_time) > 1:
                fps = self.fps_counter / (time.time() - self.fps_start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.fps_counter = 0
                self.fps_start_time = time.time()

        except Exception as e:
            print(f"Detection error: {e}")
            return frame

        return frame

    def generate_frames(self):
        """Generator function for streaming processed frames"""
        self.is_streaming = True
        cap = cv2.VideoCapture(0)
        
        try:
            while self.is_streaming:
                success, frame = cap.read()
                if not success:
                    break
                
                processed_frame = self.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        finally:
            cap.release()
            self.is_streaming = False

    def stop_streaming(self):
        """Stop the live stream"""
        self.is_streaming = False

    def process_video(self, video_path):
        """Process uploaded video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")

            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            
            output_path = f"processed_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

            cap.release()
            out.release()
            return output_path

        except Exception as e:
            print(f"Error processing video: {e}")
            raise

        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            torch.cuda.empty_cache()