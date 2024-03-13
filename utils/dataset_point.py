import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
# from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image
def pil_image_to_numpy(image, is_maks = False, index = 1):
    """Convert a PIL image to a NumPy array."""
    
    if is_maks:
#         index = 1
        image = image.resize((256, 256))
#         image = (np.array(image)==index)*1
#         image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return np.array(image)
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



def find_largest_inner_rectangle_coordinates(mask_gray):
    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    xx,yy,ww,hh = 0,0,0,0
    contours_r = contours[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) 
        if w*h > ww*hh:
            xx,yy,ww,hh = x, y, w, h
            contours_r = contour
            
            
    # 计算到轮廓的距离
    raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
    for i in range(mask_gray.shape[0]):
        for j in range(mask_gray.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_r, (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    minVal = abs(minVal)
    maxVal = abs(maxVal)
    
    return maxDistPt, int(maxVal)


class YoutubeVos(Dataset):
    def __init__(
            self,video_folder,ann_folder,motion_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):

        self.dataset = [i for i in os.listdir(video_folder)]   
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ann_folder = ann_folder
        self.heatmap = self.gen_gaussian_heatmap()
        self.motion_values_folder=motion_folder
        self.sample_size = sample_size
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
#             transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
#         self.idtransform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((196, 196)),
# #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
        
    
    
    
    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    def gen_gaussian_heatmap(self,imgSize=200):
        circle_img = np.zeros((imgSize, imgSize), np.float32)
        circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)
#         print(circle_mask)

        isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

        # Guass Map
        for i in range(imgSize):
            for j in range(imgSize):
                isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                    -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

        isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

#         isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
        return isotropicGrayscaleImage
    
    def calculate_center_coordinates(self,masks,ids, side=20):
        center_coordinates = []
        masks_list = []
        ids = random.choice(ids[1:])
        for index_mask, mask in enumerate(masks):
            new_img = np.zeros((self.sample_size, self.sample_size), np.float32)
            
            # 计算坐标的平均值，即中心坐标
#             non_zero_coordinates = np.column_stack(np.where(mask_array > 0))
#             center_coordinate = np.mean(non_zero_coordinates, axis=0)[:2].astype(np.uint8)
            
            for index in [ids]:
                mask_array = (np.array(mask)==index)*1
            
                # 找到最大距离的索引
                center_coordinate,side  = find_largest_inner_rectangle_coordinates(mask_array)
#                 center_coordinate = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)

                x1 = max(center_coordinate[0]-side,0)
                x2 = min(center_coordinate[0]+side,self.sample_size-1)
                y1 = max(center_coordinate[1]-side,0)
                y2 = min(center_coordinate[1]+side,self.sample_size-1)
                
#                 y1 = max(y,0)
#                 y2 = min(y+h,self.sample_size-1)
#                 x1 = max(x,0)
#                 x2 = min(x+w,self.sample_size-1)
                need_map = cv2.resize(self.heatmap, (x2-x1, y2-y1))
                new_img[y1:y2,x1:x2] = need_map
                
#                 if index_mask == 0:
#                     new_img = new_img + mask_array*55
                
            new_img = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            center_coordinates.append(new_img)
            masks_list.append(mask_array)
        return center_coordinates,masks_list
    
    def get_ID(self,images_list,masks_list):
        
        ID_images = []
        
        
        image = images_list[0]
        mask = masks_list[0]
        
            # 使用 findContours 函数找到轮廓
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0]) 

            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            image = image * mask

            image = image[y:y+h,x:x+w]
        except:
            pass
        
#             Id_Images = self.idtransform(Id_Images)
        image = cv2.resize(image, (196, 196))
    
    
        for i,m in zip(images_list,masks_list):
#             image = self.idtransform(Image.fromarray(image))
#             cv2.imwrite("./vis/test.jpg", image) 
            ID_images.append(image)

        return ID_images
    
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
            
            # Check if there are enough frames for both image and depth
#             if len(image_files) < 14 or len(depth_files) < 14:
#                 idx = random.randint(0, len(self.dataset) - 1)
#                 continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
    
            # Load depth frames
            mask = Image.open(os.path.join(ann_folder, depth_files[0])).convert('P')
            ids = [i for i in np.unique(mask)]
            if len(ids)==1:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
#             ids = random.choice(ids[1:])
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(ann_folder, df)).convert('P'),True,ids) for df in depth_files])
            try:
                heatmap_pixel_values, masks_list = self.calculate_center_coordinates(numpy_depth_images,ids)
            except:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
            heatmap_pixel_values = np.array(heatmap_pixel_values)
#             Id_Images = self.get_ID(numpy_images,masks_list)
            
            mask_pixel_values = numpy_to_pt(numpy_depth_images,True)
            heatmap_pixel_values = numpy_to_pt(heatmap_pixel_values,True)
#             Id_Images = numpy_to_pt(np.array(Id_Images))
            Id_Images = 0
    
            # Load motion values
            motion_values = 180
#             with open(motion_values_file, 'r') as file:
#                 motion_values = float(file.read().strip())
    
            return pixel_values, mask_pixel_values, motion_values, heatmap_pixel_values, Id_Images

        
        
    
    def __len__(self):
        return self.length
    
    def coordinates_normalize(self,center_coordinates):
        first_point = center_coordinates[0]
        center_coordinates = [one-first_point for one in center_coordinates]
        
        return center_coordinates
    
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0
    
    def normalize_sam(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return (images - torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))/torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    def __getitem__(self, idx):
        

        pixel_values, depth_pixel_values,motion_values,heatmap_pixel_values,Id_Images = self.get_batch(idx)

        pixel_values = self.normalize(pixel_values)
#         Id_Images = self.normalize_sam(Id_Images)
        
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values,
                      motion_values=motion_values,heatmap_pixel_values=heatmap_pixel_values,Id_Images=Id_Images)
        return sample




if __name__ == "__main__":
    from util import save_videos_grid

    dataset = YoutubeVos(
        video_folder = "/mmu-ocr/weijiawu/MovieDiffusion/svd-temporal-controlnet/data/ref-youtube-vos/train/JPEGImages",
        ann_folder = "/mmu-ocr/weijiawu/MovieDiffusion/svd-temporal-controlnet/data/ref-youtube-vos/train/Annotations",
        motion_folder = "",
        sample_size=256,
        sample_stride=1, sample_n_frames=16
    )
#     import pdb
#     pdb.set_trace()
    inverse_process = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        images = ((batch["pixel_values"][0].permute(0,2,3,1)+1)/2)*255
        masks = batch["depth_pixel_values"][0].permute(0,2,3,1)*255
        heatmaps = batch["heatmap_pixel_values"][0].permute(0,2,3,1)
#         Id_Images = ((batch["Id_Images"][0])*torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).permute(0,2,3,1)*255
#         center_coordinates = batch["center_coordinates"]
        
        print(batch["pixel_values"].shape)
#         print(Id_Images.shape)
        for i in range(images.shape[0]):
            image = images[i].numpy().astype(np.uint8)
#             print(Id_Images[i].shape)
#             Id_Image = inverse_process(Id_Images[i]).permute(1,2,0).numpy().astype(np.uint8)
#             Id_Image = Id_Images[i].numpy().astype(np.uint8)
#             print(Id_Image.shape)
            mask = masks[i].numpy()
            heatmap = heatmaps[i].numpy()
#             center_coordinate = center_coordinates[i][0][:2].numpy().astype(np.uint8)
            
#             print(mask.shape)
#             print(center_coordinate)
#             mask[center_coordinate[0]:center_coordinate[0]+10,center_coordinate[1]:center_coordinate[1]+10]=125 
            
            print(np.unique(mask))
#             print(Id_Image.shape)
            cv2.imwrite("./vis/image_{}.jpg".format(i), image) 
#             cv2.imwrite("./vis/Id_Image_{}.jpg".format(i), Id_Image) 
            cv2.imwrite("./vis/mask_{}.jpg".format(i), mask.astype(np.uint8)) 
            cv2.imwrite("./vis/heatmap_{}.jpg".format(i), heatmap.astype(np.uint8)) 
            cv2.imwrite("./vis/{}.jpg".format(i), heatmap.astype(np.uint8)*0.5+image*0.5) 
#             save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
        break