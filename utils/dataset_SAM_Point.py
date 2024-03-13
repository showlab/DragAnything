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

def pil_image_to_numpy(image, is_maks = False, index = 1,size=256):
    """Convert a PIL image to a NumPy array."""
    
    if is_maks:
        image = image.resize((size, size))
#         image = (np.array(image)==index)*1
#         image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return np.array(image)
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((size, size))
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

    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)
    radius = int(maxVal)

    return maxLoc, radius

# def find_largest_inner_rectangle_coordinates(mask_gray):
#     # 识别轮廓
#     contours, _ = cv2.findContours(mask_gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     xx,yy,ww,hh = 0,0,0,0
#     contours_r = contours[0]
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour) 
#         if w*h > ww*hh:
#             xx,yy,ww,hh = x, y, w, h
#             contours_r = contour
            
            
#     # 计算到轮廓的距离
#     raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
#     for i in range(mask_gray.shape[0]):
#         for j in range(mask_gray.shape[1]):
#             raw_dist[i, j] = cv2.pointPolygonTest(contours_r, (j, i), True)

#     # 获取最大值即内接圆半径，中心点坐标
#     minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
#     minVal = abs(minVal)
#     maxVal = abs(maxVal)
    
#     return maxDistPt, int(maxVal)


class YoutubeVos(Dataset):
    def __init__(
            self,video_folder,ann_folder,feature_folder,
            sample_size=512, sample_stride=4, sample_n_frames=14,
        ):

        self.dataset = [i for i in os.listdir(feature_folder)]   
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.ann_folder = ann_folder
        self.heatmap = self.gen_gaussian_heatmap()
        self.feature_folder=feature_folder
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
        
        self.idtransform = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    
    
    
    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    def gen_gaussian_heatmap(self,imgSize=200):
        circle_img = np.zeros((imgSize, imgSize), np.float32)
        circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

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
    
    def calculate_center_coordinates(self,numpy_images,masks,ids, side=20):
        center_coordinates = []
#         ids_embedding_list = []
        ids_list = {}
#         ids = random.choice(ids[1:])
        for index_mask, mask in enumerate(masks):
            new_img = np.zeros((self.sample_size, self.sample_size), np.float32)
            ids_embedding = torch.zeros((self.sample_size, self.sample_size, 1024))
            
            for index in ids[1:]:
                mask_array = (np.array(mask)==index)*1
            
                # 找到最大距离的索引
                try:
                    center_coordinate,_  = find_largest_inner_rectangle_coordinates(mask_array)
                except:
                    continue
                    print("find_largest_inner_rectangle_coordinates error")
                    
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
                
                # ID embedding
#                 if index_mask == 0:
#                     ids_list[index] = self.get_ID(numpy_images,mask_array)
                    
                    
            new_img = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            center_coordinates.append(new_img)
#             ids_embedding_list.append(mask_array)
        return center_coordinates
    
    def get_ID(self,images_list,masks_list):
        
        ID_images = []
        
        
        image = images_list[0]
        mask = masks_list
        
            # 使用 findContours 函数找到轮廓
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0]) 

            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            image = image * mask

            image = image[y:y+h,x:x+w]
        except:
            pass
            print("cv2.findContours error")

#         image = cv2.resize(image, (196, 196))
    
        image = Image.fromarray(image).convert('RGB')
        image = self.idtransform(image).unsqueeze(0).to(dtype=torch.float16)
        image.to(self.device) 
#         cls_token = self.dinov2(image, is_training=False)
        
        print(cls_token.shape)
        assert False
#         for i,m in zip(images_list,masks_list):
# #             image = self.idtransform(Image.fromarray(image))
# #             cv2.imwrite("./vis/test.jpg", image) 
#             ID_images.append(image)

        return ID_images
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('.')[0])
  
        while True:
            videoid = self.dataset[idx]
#             videoid = video_dict['videoid']
    
            preprocessed_dir = os.path.join(self.video_folder, videoid)
            ann_folder = os.path.join(self.ann_folder, videoid)
            feature_folder_file = os.path.join(self.feature_folder, videoid)
    
            if not os.path.exists(ann_folder):
                idx = random.randint(0, len(self.dataset) - 1)
                print("os.path.exists(ann_folder), error")
                continue
    
            # Sort and limit the number of image and depth files to 14
            image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)[:self.sample_n_frames]
            depth_files = sorted(os.listdir(ann_folder), key=sort_frames)[:self.sample_n_frames]
            feature_file = sorted(os.listdir(feature_folder_file), key=sort_frames)[:self.sample_n_frames]
            
            # Check if there are enough frames for both image and depth
#             if len(image_files) < 14 or len(depth_files) < 14:
#                 idx = random.randint(0, len(self.dataset) - 1)
#                 continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img)),size=self.sample_size) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
            
            # Load feature frames
            feature_images = np.array([np.array(torch.load(os.path.join(feature_folder_file, img))) for img in feature_file])
#             feature_images = torch.tensor(feature_images).permute(0, 3, 1, 2)
            feature_images = torch.from_numpy(feature_images.transpose(0, 3, 1, 2))
            
            # Load mask frames
            mask = Image.open(os.path.join(ann_folder, depth_files[0])).convert('P')
            ids = [i for i in np.unique(mask)]
            if len(ids)==1:
                idx = random.randint(0, len(self.dataset) - 1)
                print("len(ids), error")
                continue
#             ids = random.choice(ids[1:])
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(ann_folder, df)).convert('P'),True,ids,size=self.sample_size) for df in depth_files])
            heatmap_pixel_values = self.calculate_center_coordinates(numpy_images,numpy_depth_images,ids)

            heatmap_pixel_values = np.array(heatmap_pixel_values)
#             Id_Images = self.get_ID(numpy_images,masks_list)
            
            mask_pixel_values = numpy_to_pt(numpy_depth_images,True)
            heatmap_pixel_values = numpy_to_pt(heatmap_pixel_values,True)
#             Id_Images = numpy_to_pt(np.array(Id_Images))
#             Id_Images = 0
    
            # Load motion values
            motion_values = 180
#             with open(motion_values_file, 'r') as file:
#                 motion_values = float(file.read().strip())
    
            return pixel_values, mask_pixel_values, motion_values, heatmap_pixel_values, feature_images

        
        
    
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
        

        pixel_values, mask_pixel_values,motion_values,heatmap_pixel_values,feature_images = self.get_batch(idx)

        pixel_values = self.normalize(pixel_values)
        
        sample = dict(pixel_values=pixel_values, mask_pixel_values=mask_pixel_values,
                      motion_values=motion_values,heatmap_pixel_values=heatmap_pixel_values,Id_Images=feature_images)
        return sample



def load_dinov2():
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    dinov2_vitl14.eval()
#     dinov2_vitl14.requires_grad_(False)
    return dinov2_vitl14

if __name__ == "__main__":
#     from util import save_videos_grid
#     torch.multiprocessing.set_start_method('spawn')
    dino = load_dinov2()
    dino.to(dtype=torch.float16)
    
    dataset = YoutubeVos(
        video_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/JPEGImages",
        ann_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/Annotations",
        feature_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/embedding",
        sample_size=256,
        sample_stride=1, sample_n_frames=16
    )
#     import pdb
#     pdb.set_trace()
    inverse_process = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=10,)
    for idx, batch in enumerate(dataloader):
        images = ((batch["pixel_values"][0].permute(0,2,3,1)+1)/2)*255
        masks = batch["mask_pixel_values"][0].permute(0,2,3,1)*255
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