# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import json
from PIL import Image
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
from tqdm import tqdm
# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future
import cv2

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
def sort_frames(frame_name):
    return int(frame_name.split('.')[0])

def read_video_from_path_frame(path):
#     try:
#         reader = imageio.get_reader(path)
#     except Exception as e:
#         print("Error opening video file: ", e)
#         return None
    image_files = sorted(os.listdir(path), key=sort_frames)
    frames = []
    for i, im in enumerate(image_files):
        frames.append(cv2.imread(os.path.join(path,im)))
    return np.stack(frames)


def find_largest_inner_rectangle_coordinates(mask_gray):
    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)

    radius = int(maxVal)
    
    
    return maxLoc, radius

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/imgs",
        help="path to a video",
    )
    parser.add_argument(
        "--ann_path",
        default="/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/panomasks",
        help="path to a video",
    )
    parser.add_argument(
        "--save_path",
        default="/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/trajectory_CoTracker",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    if args.checkpoint is not None:
        model = CoTrackerPredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(DEFAULT_DEVICE)
    
    for iiiiidx,video_name in tqdm(enumerate(os.listdir(args.video_path))):
        save_json = os.path.join(args.save_path,video_name+".json")
#         if iiiiidx<153:
#             continue

#         if video_name!="2cdbf5f0a7":
#             continue
            
#         if os.path.exists(save_json):
#             continue
        video_path_one = os.path.join(args.video_path,video_name)
        
        video = read_video_from_path_frame(video_path_one)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video = video.to(DEFAULT_DEVICE)
        # video = video[:, :20]
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            # segm_mask=segm_mask
        )
        print("computed")
        
        # get the point in the first frame
        ann_dict = {}
        image_files = sorted(os.listdir(os.path.join(args.ann_path,video_name)), key=sort_frames)
        
        
        frames_mask = []
        for i, im in enumerate(image_files):
            frames_mask.append(cv2.imread(os.path.join(os.path.join(args.ann_path,video_name),im)))
        
        mask = np.array(Image.open(os.path.join(args.ann_path,video_name,image_files[0])))
        
#         image = np.array(Image.open(image_path))
        
        check_ids = [i for i in np.unique(np.array(mask))] 
        for index in check_ids:
            mask_array = (np.array(mask)==index)*1
            center_coordinate,_  = find_largest_inner_rectangle_coordinates(mask_array)
            ann_dict[int(index)] = center_coordinate
        
        # get the points of the all frames
        new_dict = {}
        for index in ann_dict:
            # instance point in first frame
            point2 = ann_dict[index]

            inde_min = 0
            distance_min = 1000000
            for ii,point in enumerate(pred_tracks[0][0]):
                # 计算两个点的欧氏距离
                distance = np.linalg.norm(np.array(point2) - point.cpu().numpy())
                if distance<distance_min:
                    distance_min = distance
                    inde_min = ii

            new_dict[index] = []
            for frame_id,iii in enumerate(pred_tracks[0]):
#                 xx,yy = int(iii[inde_min].cpu().numpy()[0]),int(iii[inde_min].cpu().numpy()[1])
                
#                 mask_frame = (np.array(frames_mask[frame_id])==index)*1
#                 if mask_frame[yy][xx] !=1:
#                     new_inde_min = 0
#                     new_distance_min = 1000000
#                     point2 = [xx,yy]
#                     for ii,point in enumerate(iii):
#                         distance = np.linalg.norm(np.array(point2) - point.cpu().numpy())
#                         if distance<new_distance_min:
#                             new_distance_min = distance
#                             new_inde_min = ii
#                     inde_min = new_inde_min
                    
                new_dict[index].append([int(iii[inde_min].cpu().numpy()[0]),int(iii[inde_min].cpu().numpy()[1])])

        
        with open(save_json, 'w') as json_file:
            json.dump(new_dict, json_file)

        # save a video with predicted tracks
#         seq_name = args.video_path.split("/")[-1]
#         vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
#         vis.visualize(
#             video,
#             pred_tracks,
#             pred_visibility,
#             query_frame=0 if args.backward_tracking else args.grid_query_frame,
#         )
