import torch
from PIL import Image
import cv2
import torchvision.transforms as T
import os 
import numpy as np
import torch.nn.functional as F
from dift_util import DIFT_Demo, SDFeaturizer
from torchvision.transforms import PILToTensor

from tqdm import tqdm


def load_dinov2():
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
    dinov2_vitl14.eval()
    return dinov2_vitl14

def infer_model(model, image):
    transform = T.Compose([
        T.Resize((196, 196)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).cuda()
#     cls_token = model.forward_features(image)
    cls_token = model(image, is_training=False)
    return cls_token

def sort_frames(frame_name):
    return int(frame_name.split('.')[0])

def find_largest_inner_rectangle_coordinates(mask_gray):
    # 识别轮廓
#     contours, _ = cv2.findContours(mask_gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     xx,yy,ww,hh = 0,0,0,0
#     contours_r = contours[0]
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour) 
#         if w*h > ww*hh:
#             xx,yy,ww,hh = x, y, w, h
#             contours_r = contour
            
            
    # 计算到轮廓的距离
#     raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
#     for i in range(mask_gray.shape[0]):
#         for j in range(mask_gray.shape[1]):
#             raw_dist[i, j] = cv2.pointPolygonTest(contours_r, (j, i), True)
    
    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)
    radius = int(maxVal)
    
#     # 获取最大值即内接圆半径，中心点坐标
#     minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
#     minVal = abs(minVal)
#     maxVal = abs(maxVal)
    
    return maxLoc, radius


def pil_image_to_numpy(image, is_maks = False, index = 1):
    """Convert a PIL image to a NumPy array."""
    
    if is_maks:
        image = image.resize((256, 256))
#         image = (np.array(image)==index)*1
#         image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return np.array(image)
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        return np.array(image)

def get_ID(images_list,masks_list,dinov2):
        
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

    img_embedding = infer_model(dinov2, image)


    return img_embedding

def get_dift_ID(feature_map,mask):
        
#     feature_map = feature_map * 0
    
    new_feature = []
    non_zero_coordinates = np.column_stack(np.where(mask != 0))
    for coord in non_zero_coordinates:
#         feature_map[:, coord[0], coord[1]] = 1
        new_feature.append(feature_map[:, coord[0], coord[1]])
    
    stacked_tensor = torch.stack(new_feature, dim=0)
    # 在维度0上进行平均池化
    average_pooled_tensor = torch.mean(stacked_tensor, dim=0)

    return average_pooled_tensor


def extract_dift_feature(image, dift_model):
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
           
    prompt = ''
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    dift_feature = dift_model.forward(img_tensor, prompt=prompt, up_ft_index=3,ensemble_size=8)
    return dift_feature

dinov2 = load_dinov2()
dinov2.requires_grad_(False)


model_id = 'pretrained_models/chilloutmix'
dift_model = SDFeaturizer(sd_id=model_id)


# # 加载模型
# model = torch.load("/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/embedding/2cd01cf915/1.pth")
# print(model.shape)
# assert False

dataset_type = "ref-youtube-vos"


if dataset_type == "ref-youtube-vos":
    video_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/JPEGImages"
    ann_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/Annotations"
    save_p = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/ref-youtube-vos/train/embedding_SD_512_once"
else:
    video_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/imgs"
    ann_folder = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/panomasks"
    save_p = "/mmu-ocr/weijiawu/MovieDiffusion/ShowAnything/data/VIPSeg/embedding_SD_512_once"

dataset_size = 512

dataset = [i for i in os.listdir(ann_folder)] 

for videoid in dataset:
    
    video_dir_1 = os.path.join(video_folder, videoid)
    ann_folder_1 = os.path.join(ann_folder, videoid)
    save_embedding = os.path.join(save_p, videoid)
    save_embedding_once = os.path.join(save_p, videoid+".pth")
    
#     if not os.path.exists(save_embedding):
#         print(save_embedding)
#         os.makedirs(save_embedding)
        
    image_files = sorted(os.listdir(video_dir_1), key=sort_frames)
    depth_files = sorted(os.listdir(ann_folder_1), key=sort_frames)

    #test
    keyframe_image = Image.open(os.path.join(video_dir_1, image_files[0])).convert('RGB')
    keyframe_image = keyframe_image.resize((dataset_size, dataset_size))
    keyframe_dift = extract_dift_feature(keyframe_image, dift_model=dift_model)
    
    # torch.Size([1, 320, 32, 32])
    mask = np.array(Image.open(os.path.join(ann_folder_1, depth_files[0])))
#     np.array(Image.open(os.path.join(ann_folder_1, df)))
#     mask = Image.open(os.path.join(ann_folder_1, depth_files[0])).convert('P')
    ids = [i for i in np.unique(mask)]    
    numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(ann_folder_1, df)),True,ids) for df in depth_files])
    
    ids_list = {}
    for index_mask, mask in tqdm(enumerate(numpy_depth_images)):
        ids_embedding = torch.ones((dataset_size, dataset_size, 320))

        # 判断文件是否存在
#         if os.path.exists(os.path.join(save_embedding, '{}.pth'.format(index_mask))) and index_mask!=0:
#             continue
        
        for index in ids:
            mask_array = (np.array(mask)==index)*1
            
            try:
                center_coordinate,_  = find_largest_inner_rectangle_coordinates(mask_array)
            except:
                continue
                print("find_largest_inner_rectangle_coordinates error")
            

            circle_img = np.zeros((dataset_size, dataset_size), np.float32)
            circle_mask = cv2.circle(circle_img, (center_coordinate[0],center_coordinate[1]), 20, 1, -1)
            
            
            # ID embedding
            if index_mask == 0:
                # diffusion feature
                mask_32 = cv2.resize(mask_array.astype(np.uint8),(int(dataset_size/8),int(dataset_size/8)))
                if len(np.column_stack(np.where(mask_32 != 0)))==0:
                    continue
                    
                id_feature = get_dift_ID(keyframe_dift[0],mask_32)
                ids_list[index] = id_feature
                
            else:
                try:
                    id_feature = ids_list[index]
                except:
                    print("index error")
                    continue
            
#             获取非零像素的坐标
#             non_zero_coordinates = np.column_stack(np.where(circle_mask != 0))
#             for coord in non_zero_coordinates:
#                 ids_embedding[coord[0], coord[1]] = id_feature
        
        torch.save(ids_list, save_embedding_once)
        
        # only extract the feature of the first frame
        break
        
#         ids_embedding = F.avg_pool3d(ids_embedding, kernel_size=(2, 1, 1), stride=(8, 1, 1))      

        ids_embedding = F.avg_pool1d(ids_embedding, kernel_size=2, stride=2)
        torch.save(ids_embedding, os.path.join(save_embedding, '{}.pth'.format(index_mask)))
            
