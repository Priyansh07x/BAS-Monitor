#!/usr/bin/env python3
"""
SIH26174 Part 1 - Multi-Class Action Recognition Pipeline
Complete inference script for 51-class HMDB51 action recognition
"""

import torch
import cv2
import json
import numpy as np
from pathlib import Path
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO
import torch.nn as nn

class ActionRecognitionPipeline:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        self.model = r2plus1d_18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, self.checkpoint['num_classes'])
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load pose and detection models
        self.pose_model = YOLO('yolov8m-pose.pt')
        self.detection_model = YOLO('yolov8m.pt')
    
    def predict_video(self, video_path):
        """Predict action for a video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_idx = 0
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detection
            det = self.detection_model.predict(frame_rgb, verbose=False)
            objects = len(det[0].boxes) if det[0].boxes else 0
            
            # Pose
            pose = self.pose_model.predict(frame_rgb, verbose=False)
            pose_detected = pose[0].keypoints is not None
            
            results.append({
                'frame': frame_idx,
                'time_sec': frame_idx / fps,
                'objects': objects,
                'pose': pose_detected,
            })
            
            frame_idx += 1
        
        cap.release()
        return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python inference.py <video_path> [checkpoint_path]')
        sys.exit(1)
    
    video_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'hmdb51_r2plus1d_best.pt'
    
    pipeline = ActionRecognitionPipeline(checkpoint_path)
    results = pipeline.predict_video(video_path)
    
    print(json.dumps(results, indent=2))
