from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def build_action_model(num_classes=2):
    model = r2plus1d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    return model


def load_action_model(package_dir, device=None):
    package_dir = Path(package_dir)
    config = load_config(package_dir / "runtime_config.json")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(package_dir / config["checkpoint_filename"], map_location=device)
    model = build_action_model(num_classes=len(config["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, device


def read_video_rgb(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames, fps


def make_windows(num_frames, clip_length, stride):
    if num_frames < clip_length:
        return [np.linspace(0, num_frames - 1, clip_length).round().astype(int)]
    starts = list(range(0, num_frames - clip_length + 1, stride))
    final_start = num_frames - clip_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [np.arange(s, s + clip_length, dtype=int) for s in starts]


def preprocess_clip(frames_rgb, config):
    image_size = int(config["image_size"])
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    processed = []
    for frame in frames_rgb:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - mean) / std
        processed.append(np.transpose(arr, (2, 0, 1)))
    return torch.from_numpy(np.stack(processed, axis=1)).float()


@torch.no_grad()
def predict_action_timeline(video_path, package_dir, max_frames=None):
    model, config, device = load_action_model(package_dir)
    frames, fps = read_video_rgb(video_path, max_frames=max_frames)
    windows = make_windows(len(frames), int(config["clip_length"]), int(config["window_stride_frames"]))
    class_to_idx = config["class_to_idx"]
    idx_to_class = {int(k): v for k, v in config["idx_to_class"].items()}
    amp_enabled = device.type == "cuda"

    rows = []
    for batch_start in range(0, len(windows), 8):
        batch_windows = windows[batch_start:batch_start + 8]
        clips = []
        for inds in batch_windows:
            clips.append(preprocess_clip([frames[int(i)] for i in inds], config))
        batch = torch.stack(clips, dim=0).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            probs = torch.softmax(model(batch), dim=1)
        probs_np = probs.detach().cpu().numpy()
        preds_np = probs_np.argmax(axis=1)
        for local_i, inds in enumerate(batch_windows):
            pred_id = int(preds_np[local_i])
            rows.append({
                "window_id": batch_start + local_i,
                "start_frame": int(inds[0]),
                "end_frame": int(inds[-1]),
                "center_time_sec": float(np.mean(inds)) / fps if fps > 0 else None,
                "raw_catch_prob": float(probs_np[local_i, class_to_idx["catch"]]),
                "raw_not_catch_prob": float(probs_np[local_i, class_to_idx["not_catch"]]),
                "raw_pred_label": idx_to_class[pred_id],
            })

    df = pd.DataFrame(rows)
    df["smooth_catch_prob"] = df["raw_catch_prob"].ewm(alpha=float(config["ema_alpha"]), adjust=False).mean()
    active = False
    states = []
    for prob in df["smooth_catch_prob"].tolist():
        if not active and prob >= float(config["action_on_threshold"]):
            active = True
        elif active and prob <= float(config["action_off_threshold"]):
            active = False
        states.append(active)
    df["catch_state"] = states
    return df
