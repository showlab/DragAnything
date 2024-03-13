import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
# from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image
def pil_image_to_numpy(image, is_maks = False):
    """Convert a PIL image to a NumPy array."""
    
    if is_maks:
        image = image.resize((256, 256))
        image = (np.array(image)==1)*1
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
            
#         print(np.unique(np.array(image)))
        return image
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        return np.array(image)

def numpy_to_pt(images: np.ndarray, is_mask=False) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    if is_mask:
        return images.float() 
    else:
        return images.float() / 255


class WebVid10M(Dataset):
    def __init__(
            self,video_folder,ann_folder,motion_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):

        self.dataset = [i for i in os.listdir(video_folder)]
#         self.dataset = ["cce03c2a9b"]    
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ann_folder = ann_folder
        self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
#             transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('.')[0])
    

    
        while True:
            videoid = self.dataset[idx]
#             videoid = video_dict['videoid']
    
            preprocessed_dir = os.path.join(self.video_folder, videoid)
            ann_folder = os.path.join(self.ann_folder, videoid)
            motion_values_file = os.path.join(self.motion_values_folder, videoid, videoid + "_average_motion.txt")
    
            if not os.path.exists(ann_folder):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Sort and limit the number of image and depth files to 14
            image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)[:14]
            depth_files = sorted(os.listdir(ann_folder), key=sort_frames)[:14]
            
#             print(image_files)
#             print(depth_files)
            # Check if there are enough frames for both image and depth
#             if len(image_files) < 14 or len(depth_files) < 14:
#                 idx = random.randint(0, len(self.dataset) - 1)
#                 continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
    
            # Load depth frames
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(ann_folder, df)).convert('P'),True) for df in depth_files])
            # 
            mask_pixel_values = numpy_to_pt(numpy_depth_images,True)
#             print(np.unique(depth_pixel_values))
            
            
            # Load motion values
            motion_values = 180
#             with open(motion_values_file, 'r') as file:
#                 motion_values = float(file.read().strip())
    
            return pixel_values, mask_pixel_values, motion_values

        
        
    
    def __len__(self):
        return self.length
    
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0
    
    def __getitem__(self, idx):
        
        #while True:
           # try:
        pixel_values, depth_pixel_values,motion_values = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)
#         print()
        pixel_values = self.normalize(pixel_values)
    
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values,motion_values=motion_values)
        return sample




if __name__ == "__main__":
    from util import save_videos_grid

    dataset = WebVid10M(
        video_folder = "/mmu-ocr/weijiawu/MovieDiffusion/svd-temporal-controlnet/data/ref-youtube-vos/train/JPEGImages",
        ann_folder = "/mmu-ocr/weijiawu/MovieDiffusion/svd-temporal-controlnet/data/ref-youtube-vos/train/Annotations",
        motion_folder = "",
        sample_size=256,
        sample_stride=4, sample_n_frames=16
    )
#     import pdb
#     pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        images = batch["pixel_values"][0].permute(0,2,3,1)*255
        masks = batch["depth_pixel_values"][0].permute(0,2,3,1)
        
        print(batch["pixel_values"].shape)

        for i in range(images.shape[0]):
            image = images[i].numpy().astype(np.uint8)
            mask = masks[i].numpy().astype(np.uint8)*255
            print(np.unique(mask))
            cv2.imwrite("./vis/image_{}.jpg".format(i), image) 
            cv2.imwrite("./vis/mask_{}.jpg".format(i), mask) 
#             save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
        break