# SVD: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
# SVD 1.1:  https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_DragAnything.py \
 --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="model_out/ShowAnything-2024.3.4-Gaussian-SD1.5_25frames_VIPSeg_Size576_320" \
 --video_folder="./data/VIPSeg/imgs" \
 --mask_folder="./data/VIPSeg/panomasks" \
 --feature_folder="./data/VIPSeg/embedding_SD_512_once" \
 --validation_image_folder="./validation_demo/cce03c2a9b_Image" \
 --validation_control_folder="./validation_demo/cce03c2a9b_Mask" \
 --width=576 \
 --height=320 \
 --num_frames=25\
 --learning_rate=2e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=500 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=2000 \
 --gradient_checkpointing
