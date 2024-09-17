import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def uniformly_sample_frames(frames, num_samples):
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
    sampled_frames = [frames[i] for i in indices]
    return sampled_frames

def delete_first_and_last_frames(frames, num_frames=10):
    frames = frames[num_frames:-num_frames]
    return frames

def visualize_frames(frames):
    # fig, axes = plt.subplots(1, len(frames), figsize=(20, 5))
    # for idx, frame in enumerate(frames):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     axes[idx].imshow(frame_rgb)
    #     axes[idx].axis('off')
    # plt.show()

    num_frames = len(frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    for idx, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(frame_rgb)
        axes[idx].axis('off')
    plt.subplots_adjust(wspace=0.01)  # Reduce the space between frames
    plt.show()

# Path to your video file
video_path = r'G:\video\Sep12-16\Sep15\E15_H2_-1m_b.mp4'

# Read the video
frames = read_video(video_path)
print(len(frames))

# Number of frames to sample uniformly
num_samples = 5
#frames = delete_first_and_last_frames(frames, num_frames=10)\
frames = frames[250:300]
sampled_frames = uniformly_sample_frames(frames, num_samples)

# Convert frames to PyTorch tensors (if needed)
#sampled_frames_tensors = [torch.tensor(frame) for frame in frames]

# Visualize the sampled frames
visualize_frames(sampled_frames)
