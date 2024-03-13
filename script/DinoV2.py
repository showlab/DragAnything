import torch
from PIL import Image
import cv2
import torchvision.transforms as T

# dinov2_vitl14
# dinov2_vitg14

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
    cls_token = model.forward_features(image)
    return cls_token

dinov2 = load_dinov2()
dinov2.requires_grad_(False)
image = "./validation_demo/3373891cdc_Image/1704429543488.jpg"
image = Image.open(image).convert('RGB')
# image = image.resize((64,64))
img_embedding = infer_model(dinov2, image)
print(img_embedding["x_norm_patchtokens"].shape,img_embedding["x_norm_clstoken"].shape)