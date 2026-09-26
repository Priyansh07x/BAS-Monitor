import cv2
import os
import sys

# Get absolute paths dynamically so the script runs from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from backend.ai.inference_pipeline import InferencePipeline

def test_ai():
    print("Initializing AI Pipeline...")
    pipeline = InferencePipeline()
    
    # Absolute path to the sample video
    video_path = os.path.join(ROOT_DIR, "data", "videos", "REC_20260905_165751.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Could not find {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    print(f"Processing video: {video_path}")
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < 10:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        
        result = pipeline.process_frame(frame, annotate=True)
        
        pose_count = 1 if result["pose"] else 0
        hand_count = len(result["hands"]) if result["hands"] else 0
        obj_count = len(result["objects"]) if result["objects"] else 0
        
        print(f"Frame {frame_count}: Found {pose_count} poses, {hand_count} hands, {obj_count} objects. Action: {result['action']}")
        
        if frame_count == 10:
            # Save the annotated frame to tests/test_results/
            output_dir = os.path.join(SCRIPT_DIR, "test_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "ai_test_output.jpg")
            
            cv2.imwrite(output_path, result["annotated_frame"])
            print(f"\n✅ Saved visual output to: {output_path}")
            
    cap.release()
    pipeline.release()

if __name__ == "__main__":
    test_ai()
