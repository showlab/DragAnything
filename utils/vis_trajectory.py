import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import os 
from PIL import Image

# image_path = "./VIPSeg_Video_Generation_Test/imgs/443_hPrsXtTg3lM"
# trajectory = "./VIPSeg_Video_Generation_Test/trajectory_CoTracker/443_hPrsXtTg3lM.json"
save_image = "./VIPSeg_Video_Generation_Test/vis"


# image_path_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/imgs"
# trajectory_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/test_traject"
# save_image_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/trajectory_vis"

image_path_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/Prediction_Model/DragAnything14frames_OriginalSize"
trajectory_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/Prediction_Model/trajectory_1024_CoTracker_DragAnything14frames_OriginalSize1"
save_image_root = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/VIPSeg_Video_Generation_Test/trajectory_vis"



def sort_frames(frame_name):
    return int(frame_name.split('.')[0])

for video_name in os.listdir(image_path_root):
    image_path = os.path.join(image_path_root,video_name)
    trajectory = os.path.join(trajectory_root,video_name+".json")
    save_image = os.path.join(save_image_root,video_name+".gif")
    
    with open(trajectory, 'r') as json_file:
        data = json.load(json_file)

    image_files = sorted(os.listdir(image_path), key=sort_frames)

    pil_images = []
    for idx,images in enumerate(image_files):
        image = cv2.imread(os.path.join(image_path,images))
        for line in data:
            line_data = data[line][:(idx+1)]
            print(line_data)
            if len(line_data)>=2:
                for i in range(len(line_data)-1):
                    cv2.line(image, line_data[i], line_data[i+1], (0, 255, 0), 3)

        cv2.imwrite(os.path.join(save_image,images),image)
        pil_images.append(Image.fromarray(image))

    pil_images[0].save(save_image, save_all=True, append_images=pil_images[1:], loop=0, duration=110)
