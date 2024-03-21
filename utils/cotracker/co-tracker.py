import torch
from cotracker.utils.visualizer import Visualizer

# Download the video
url = 'https://github.com/facebookresearch/co-tracker/blob/main/assets/apple.mp4'

import imageio.v3 as iio

frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1


vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)