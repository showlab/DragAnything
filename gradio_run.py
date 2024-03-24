import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter

import torch
import datetime
import numpy as np
import uuid
from pipeline.pipeline_svd_DragAnything import StableVideoDiffusionPipeline
from models.DragAnything import DragAnythingSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re 
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as T
import torch.nn.functional as F
from utils.dift_util import DIFT_Demo, SDFeaturizer
from torchvision.transforms import PILToTensor
import json
from utils_drag import *
from scipy.interpolate import interp1d, PchipInterpolator
from segment_anything import sam_model_registry, SamPredictor
import imageio
from moviepy.editor import *

print("gr file",gr.__file__)

color_list = []
for i in range(20):
    color = np.concatenate([np.random.random(4)*255], axis=0)
    color_list.append(color)



output_dir = "./outputs"
ensure_dirname(output_dir)


# SAM
sam_checkpoint = "./script/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

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

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points

def visualize_drag_v2(background_image_path, splited_tracks, width, height):
    trajectory_maps = []
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 5, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer

def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    return isotropicGrayscaleImage

def extract_dift_feature(image, dift_model):
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.open(image).convert('RGB')
           
    prompt = ''
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    dift_feature = dift_model.forward(img_tensor, prompt=prompt, up_ft_index=3,ensemble_size=8)
    return dift_feature

def get_condition(target_size=(512 , 512), points=None, original_size=(512 , 512), args="", first_frame=None, is_mask = False, side=20,model_id=None):
    images = []
    vis_images = []
    heatmap = gen_gaussian_heatmap()
    
    original_size = (original_size[1],original_size[0])
    size = (target_size[1],target_size[0])
    latent_size = (int(target_size[1]/8), int(target_size[0]/8))
    
    
    dift_model = SDFeaturizer(sd_id=model_id)
    keyframe_dift = extract_dift_feature(first_frame, dift_model=dift_model)
    
    ID_images=[]
    ids_list={}

    mask_list = []
    trajectory_list = []
    radius_list = []
    
    for index,point in enumerate(points):
        mask_name = output_dir+"/"+"mask_{}.jpg".format(index+1)
        trajectories = [[int(i[0]),int(i[1])] for i in point]
        trajectory_list.append(trajectories)
        
        first_mask = (cv2.imread(mask_name)/255).astype(np.uint8)
        first_mask = cv2.cvtColor(first_mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mask_list.append(first_mask)
        
        mask_322 = cv2.resize(first_mask.astype(np.uint8),(int(target_size[1]), int(target_size[0])))
        _, radius = find_largest_inner_rectangle_coordinates(mask_322)
        radius_list.append(radius)    
    

    for idxx,point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        ids_embedding = torch.zeros((target_size[0], target_size[1], 320))
        
        if idxx>= args["frame_number"]:
            break
            
        for cc,(mask,trajectory,radius) in enumerate(zip(mask_list,trajectory_list,radius_list)):
            
            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius,50)

            # ID embedding
            if idxx == 0:
                # diffusion feature
                mask_32 = cv2.resize(mask.astype(np.uint8),latent_size)
                if len(np.column_stack(np.where(mask_32 != 0)))==0:
                    continue
                ids_list[cc] = get_dift_ID(keyframe_dift[0],mask_32)

                id_feature = ids_list[cc]
            else:
                id_feature = ids_list[cc]

            circle_img = np.zeros((target_size[0], target_size[1]), np.float32)
            circle_mask = cv2.circle(circle_img, (center_coordinate[0],center_coordinate[1]), side, 1, -1)
                      
            y1 = max(center_coordinate[1]-side,0)
            y2 = min(center_coordinate[1]+side,target_size[0]-1)
            x1 = max(center_coordinate[0]-side,0)
            x2 = min(center_coordinate[0]+side,target_size[1]-1)
            
            if x2-x1>3 and y2-y1>3:
                need_map = cv2.resize(heatmap, (x2-x1, y2-y1))
                new_img[y1:y2,x1:x2] = need_map.copy()
                
                if cc>=0:
                    vis_img[y1:y2,x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1],trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_)-1):
                            cv2.line(vis_img,(trajectory_[itt][0],trajectory_[itt][1]),(trajectory_[itt+1][0],trajectory_[itt+1][1]),(255,255,255),3)

            non_zero_coordinates = np.column_stack(np.where(circle_mask != 0))
            for coord in non_zero_coordinates:
                ids_embedding[coord[0], coord[1]] = id_feature[0]
        
        ids_embedding = F.avg_pool1d(ids_embedding, kernel_size=2, stride=2)
        img = new_img

        # Ensure all images are in RGB format
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
        # Convert the numpy array to a PIL image
        pil_img = Image.fromarray(img)
        images.append(pil_img)
        vis_images.append(Image.fromarray(vis_img))
        ID_images.append(ids_embedding)
    return images,ID_images,vis_images

def find_largest_inner_rectangle_coordinates(mask_gray):

    refine_dist = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(refine_dist)
    radius = int(maxVal)

    return maxLoc, radius

def save_gifs_side_by_side(batch_output, validation_control_images,output_folder,name = 'none', target_size=(512 , 512),duration=200):

    flattened_batch_output = batch_output
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img,target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    
#     validation_control_images = validation_control_images*255 validation_images, 
    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):
        
#         if idx==0:
#             continue

        gif_path = os.path.join(output_folder.replace("vis_gif.gif",""), f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[0].n_frames):
            combined_frame = None
            
                
            for gif in gifs:
                
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
        print(gifs[0].info['duration'])
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)
        

    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = output_folder
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # Clean up temporary GIFs
    for gif_path in gif_paths:
        os.remove(gif_path)

    return combined_gif_path

# Define functions
def validate_and_convert_image(image, target_size=(512 , 512)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

class Drag:
    def __init__(self, device, args, height, width, model_length):
        self.device = device

        self.controlnet = controlnet = DragAnythingSDVModel.from_pretrained(args["DragAnything"])
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"],subfolder="unet")
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args["pretrained_model_name_or_path"],controlnet=self.controlnet,unet=unet)
        self.pipeline.enable_model_cpu_offload()

        self.height = height
        self.width = width
        self.args = args
        self.model_length = model_length


    def run(self, first_frame_path, tracking_points, inference_batch_size, motion_bucket_id):
        original_width, original_height=576, 320

        input_all_points = tracking_points.constructor_args['value']
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]
        
        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        for idx,splited_track in enumerate(resized_all_points):
            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[:self.model_length]
            resized_all_points[idx]=splited_track
        
        validation_image = Image.open(first_frame_path).convert('RGB')
        width, height = validation_image.size
        validation_image = validation_image.resize((self.width, self.height))
        validation_control_images,ids_embedding,vis_images = get_condition(target_size=(self.args["height"] , self.args["width"]),points = resized_all_points,
                                                                       original_size=(self.height , self.width),
                                                                       args = self.args,first_frame = validation_image,
                                                                      side=100,model_id=args["model_DIFT"])
        ids_embedding = torch.stack(ids_embedding, dim=0).permute(0, 3, 1, 2)
        
        # Inference and saving loop
        video_frames = self.pipeline(validation_image, validation_control_images[:self.model_length], decode_chunk_size=8,num_frames=self.model_length,motion_bucket_id=motion_bucket_id,controlnet_cond_scale=1.0,height=self.height,width=self.width,ids_embedding=ids_embedding[:self.model_length]).frames
        
        vis_images = [cv2.applyColorMap(np.array(img).astype(np.uint8), cv2.COLORMAP_JET) for img in vis_images]
        vis_images = [cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB) for img in vis_images]
    
        vis_images = [Image.fromarray(img) for img in vis_images]
    
        video_frames = [img for sublist in video_frames for img in sublist]
        val_save_dir = output_dir+"/"+"vis_gif.gif"
        save_gifs_side_by_side(video_frames, vis_images[:self.model_length],val_save_dir,target_size=(self.width,self.height),duration=110)
#         clip = Image.open(val_save_dir)
#         print(clip.size)
        return val_save_dir

args = {
        "pretrained_model_name_or_path": "stabilityai/stable-video-diffusion-img2vid",
        "DragAnything":"./model_out/DragAnything",
        "model_DIFT":"./utils/pretrained_models/chilloutmix",
        
        "validation_image": "./validation_demo/Demo/ship_@",
        
        "output_dir": "./validation_demo",
        "height":   320,
        "width":  576,
        
        "frame_number": 14
        # cant be bothered to add the args in myself, just use notepad
    }

with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">DragAnything 1.0</h1><br>""")

    gr.Markdown("""Gradio Demo for <a href='https://arxiv.org/abs/2403.07420'><b>DragAnything: Motion Control for Anything using Entity Representation</b></a>. The template is inspired by DragNUWA.""")

    gr.Image(label="DragAnything", value="assets/output.gif")

    gr.Markdown("""## Usage: <br>
                1. Upload an image via the "Upload Image" button.<br>
                2. Draw some drags.<br>
                    2.1. Click "Select Area with SAM" to select the area that you want to control.<br>
                    2.2. Click "Add Drag Trajectory" to add the motion trajectory.<br>
                    2.3. You can click several points which forms a path.<br>
                    2.4. Click "Delete last drag" to delete the whole lastest path.<br>
                    2.5. Click "Delete last step" to delete the lastest clicked control point.<br>
                3. Animate the image according the path with a click on "Run" button. <br>""")
    
#     device, args, height, width, model_length
    
    DragAnything = Drag("cuda:0", args, 320, 576, 14)
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    
    flag_points = gr.State()
    
    def reset_states(first_frame_path, tracking_points):
        first_frame_path = gr.State()
        tracking_points = gr.State([])
        return first_frame_path, tracking_points

    def preprocess_image(image):

        image_pil = image2pil(image.name)
        
            
        raw_w, raw_h = image_pil.size
        resize_ratio = max(576/raw_w, 320/raw_h)
        image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        image_pil = transforms.CenterCrop((320, 576))(image_pil.convert('RGB'))

        first_frame_path = os.path.join(output_dir, f"first_frame_{str(uuid.uuid4())[:4]}.png")
        
        image_pil.save(first_frame_path)

        return first_frame_path, first_frame_path, gr.State([])

    def add_drag(tracking_points):
        tracking_points.constructor_args['value'].append([])
        return tracking_points,0
        
    def re_add_drag(tracking_points):
        tracking_points.constructor_args['value'][-1]=[]
        return tracking_points,1
    
    
    def delete_last_drag(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map
    
    def delete_last_step(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'][-1].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map
    
    def add_tracking_points(tracking_points, first_frame_path, flag_points, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        tracking_points.constructor_args['value'][-1].append(evt.index)
        
        if flag_points==1:
            transparent_background = Image.open(first_frame_path).convert('RGBA')
            
            
            w, h = transparent_background.size
            transparent_layer = 0
            for idx,track in enumerate(tracking_points.constructor_args['value']):
                mask = cv2.imread(output_dir+"/"+"mask_{}.jpg".format(idx+1))
                color = color_list[idx+1]
                transparent_layer =  mask[:,:,0].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            
                if len(track) > 1:
                    for i in range(len(track)-1):
                        start_point = track[i]
                        end_point = track[i+1]
                        vx = end_point[0] - start_point[0]
                        vy = end_point[1] - start_point[1]
                        arrow_length = np.sqrt(vx**2 + vy**2)
                        if i == len(track)-2:
                            cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                        else:
                            cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
                else:
                    cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)
            transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            alpha_coef = 0.99
            im2_data = transparent_layer.getdata()
            new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
            transparent_layer.putdata(new_im2_data)
                
            trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        else:
            transparent_background = Image.open(first_frame_path).convert('RGBA')
            w, h = transparent_background.size
            
            
            input_point = []
            input_label = []
            for track in tracking_points.constructor_args['value'][-1]:
                input_point.append([track[0],track[1]])
                input_label.append(1)
            image = cv2.imread(first_frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            

            input_point = np.array(input_point)
            input_label = np.array(input_label)

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            cv2.imwrite(output_dir+"/"+"mask_{}.jpg".format(len(tracking_points.constructor_args['value'])),masks[1]*255)
            
            
            color = color_list[len(tracking_points.constructor_args['value'])]
            transparent_layer =  masks[1].reshape(h, w, 1) * color.reshape(1, 1, -1)
            
            transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            alpha_coef = 0.99
            im2_data = transparent_layer.getdata()
            new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
            transparent_layer.putdata(new_im2_data)
            
#             transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map

    with gr.Row():
        with gr.Column(scale=1):
            image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
            select_area_button = gr.Button(value="Select Area with SAM")
            add_drag_button = gr.Button(value="Add New Drag Trajectory")
            reset_button = gr.Button(value="Reset")
            run_button = gr.Button(value="Run")
            delete_last_drag_button = gr.Button(value="Delete last drag")
            delete_last_step_button = gr.Button(value="Delete last step")

        with gr.Column(scale=7):
            with gr.Row():
                with gr.Column(scale=6):
                    input_image = gr.Image(label="SAM mask",
                                        interactive=True,
                                        height=320,
                                        width=576,)
    
    with gr.Row():
        with gr.Column(scale=1):
            inference_batch_size = gr.Slider(label='Inference Batch Size', 
                                             minimum=1, 
                                             maximum=1, 
                                             step=1, 
                                             value=1)
            
            motion_bucket_id = gr.Slider(label='Motion Bucket', 
                                             minimum=1, 
                                             maximum=180, 
                                             step=1, 
                                             value=100)

        with gr.Column(scale=5):
            output_video =  gr.Image(label="Output Video",
                                    height=320,
                                    width=1152,)

    with gr.Row():
        gr.Markdown("""
            ## Citation
            ```bibtex
            @article{wu2024draganything,
              title={DragAnything: Motion Control for Anything using Entity Representation},
              author={Wu, Wejia and Li, Zhuang and Gu, Yuchao and Zhao, Rui and He, Yefei and Zhang, David Junhao and Shou, Mike Zheng and Li, Yan and Gao, Tingting and Zhang, Di},
              journal={arXiv preprint arXiv:2403.07420},
              year={2024}
            }
            ```
            """)
    
    print("debug 1")
            
    image_upload_button.upload(preprocess_image, image_upload_button, [input_image, first_frame_path, tracking_points])
    
    select_area_button.click(add_drag, tracking_points, [tracking_points,flag_points])
    
    add_drag_button.click(re_add_drag, tracking_points, [tracking_points,flag_points])
    
    delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path], [tracking_points, input_image])

    delete_last_step_button.click(delete_last_step, [tracking_points, first_frame_path], [tracking_points, input_image])

    reset_button.click(reset_states, [first_frame_path, tracking_points], [first_frame_path, tracking_points])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path,flag_points], [tracking_points, input_image])

    run_button.click(DragAnything.run, [first_frame_path, tracking_points, inference_batch_size, motion_bucket_id], output_video)

    demo.queue().launch(server_name="0.0.0.0",share=True)
#     demo.launch(server_name="0.0.0.0",share=True)
