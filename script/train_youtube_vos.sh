# SVD: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
# SVD 1.1:  https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_PointNet.py \
 --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="model_out/ShowAnything-2024.1.23-Gaussian-SD1.5-weightloss_20frames" \
 --video_folder="./data/ref-youtube-vos/train/JPEGImages" \
 --depth_folder="./data/ref-youtube-vos/train/Annotations" \
 --motion_folder="./data/ref-youtube-vos/train/embedding_SD_nobackground_512" \
 --validation_image_folder="./validation_demo/cce03c2a9b_Image" \
 --validation_control_folder="./validation_demo/cce03c2a9b_Mask" \
 --width=512 \
 --height=512 \
 --num_frames=20\
 --learning_rate=1e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=500 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=2000 \
 --validation_steps=500 \
 --gradient_checkpointing
