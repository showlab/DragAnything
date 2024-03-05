# DragAnything

### <div align="center"> DragAnything: Motion Control for Anything using Entity Representation <div> 

<div align="center">
  <a href="https://weijiawu.github.io/ParaDiffusionPage/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2311.14284"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
</div>

<p align="center">
<img src="./assets/1709656085862.jpg" width="800px"/>  
<br>
</p>


## :notes: **Updates**

- [ ] Nov. 15, 2023. Release the train code in **three months**.
- [x] Mar. 10, 2024. Rep initialization.


---

## 🐱 Abstract
<font color="red">ParaDiffusion</font> an information-enriched diffusion model for paragraph-to-image generation task, which delves into the transference of the extensive semantic comprehension capabilities of large language models to the task of image generation. At its core is using a large language model (e.g., Llama V2) to encode long-form text, followed by fine-tuning with LORA to align the text-image feature spaces in the generation task. A high-quality paragraph-image pair dataset, namely ParaImage is proposed to facilitate the training of long-text semantic alignment.

---
## User-Trajectory Interaction with SAM
<table class="center">
<tr>
      <td style="text-align:center;"><b>Input Image</b></td>
  <td style="text-align:center;"><b>Drag point with SAM</b></td>
    <td style="text-align:center;"><b>2D Gaussian Trajectory</b></td>
      <td style="text-align:center;"><b>Generated Video</b></td>
</tr>
<tr>
  <td><img src="./assets/1709660422197.jpg" width="177" height="100"></td>
  <td><img src="./assets/1709660459944.jpg" width="177" height="100"></td>
  <td><img src="./assets/image28 (3).gif" width="177" height="100"></td>              
  <td><img src="./assets/image28 (2).gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709660422197.jpg" width="177" height="100"></td>
  <td><img src="./assets/1709660471568.jpg" width="177" height="100"></td>
  <td><img src="./assets/image2711.gif" width="177" height="100"></td>              
  <td><img src="./assets/image27 (1)1.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709660422197.jpg" width="177" height="100"></td>
  <td><img src="./assets/1709660965701.jpg" width="177" height="100"></td>
  <td><img src="./assets/image29111.gif" width="177" height="100"></td>              
  <td><img src="./assets/image29 (1)1.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709660422197.jpg" width="177" height="100"></td>
  <td><img src="./assets/1709661150250.jpg" width="177" height="100"></td>
  <td><img src="./assets/image30 (1)1.gif" width="177" height="100"></td>              
  <td><img src="./assets/image3011.gif" width="177" height="100"></td>
</tr>

</table>


## Comparison with DragNUWA
<table class="center">
<tr>
      <td style="text-align:center;"><b>Model</b></td>
  <td style="text-align:center;"><b>Input Image and Drag</b></td>
    <td style="text-align:center;"><b>Generated Video</b></td>
      <td style="text-align:center;"><b>Visualization for Pixel Motion</b></td>
</tr>
<tr>
  <td style="text-align:center;"><b>DragNUWA</b></td>
  <td><img src="./assets/1709661872632.jpg" width="177" height="100"></td>
  <td><img src="./assets/image63111.gif" width="177" height="100"></td>              
  <td><img src="./assets/image6411.gif" width="177" height="100"></td>
</tr>


</table>



## More Demo


<table class="center">
<tr>
  <td style="text-align:center;"><b>Drag point with SAM</b></td>
  <td style="text-align:center;"><b>2D Gaussian</b></td>
    <td style="text-align:center;"><b>Generated Video</b></td>
      <td style="text-align:center;"><b>Visualization for Pixel Motion</b></td>
</tr>
<tr>
  <td><img src="./assets/1709656550343.jpg" width="177" height="100"></td>
  <td><img src="./assets/image188.gif" width="177" height="100"></td>
  <td><img src="./assets/image190.gif" width="177" height="100"></td>              
  <td><img src="./assets/image189.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709657635807.jpg" width="177" height="100"></td>
  <td><img src="./assets/image187 (1).gif" width="177" height="100"></td>
  <td><img src="./assets/image186.gif" width="177" height="100"></td>              
  <td><img src="./assets/image185.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709658516913.jpg" width="177" height="100"></td>
  <td><img src="./assets/image158.gif" width="177" height="100"></td>
  <td><img src="./assets/image159.gif" width="177" height="100"></td>              
  <td><img src="./assets/image160.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709658781935.jpg" width="177" height="100"></td>
  <td><img src="./assets/image163.gif" width="177" height="100"></td>
  <td><img src="./assets/image161.gif" width="177" height="100"></td>              
  <td><img src="./assets/image162.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709659276722.jpg" width="177" height="100"></td>
  <td><img src="./assets/image165.gif" width="177" height="100"></td>
  <td><img src="./assets/image167.gif" width="177" height="100"></td>              
  <td><img src="./assets/image166.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709659787625.jpg" width="177" height="100"></td>
  <td><img src="./assets/image172.gif" width="177" height="100"></td>
  <td><img src="./assets/Our_Motorbike_cloud_floor.gif" width="177" height="100"></td>              
  <td><img src="./assets/image171.gif" width="177" height="100"></td>
</tr>


</table>


##  Various Motion Control 


## 🔧 Dependencies and Dataset Prepare

### Dependencies
- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)

```Shell
git clone https://github.com/Showlab/DragAnything.git
cd DragAnything

conda create -n DragAnything python=3.8
conda activate DragAnything
pip install -r environment.txt
```

### Dataset Prepare

Download [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) and [Youtube-VOS](https://youtube-vos.org/) to the ```./data``` directory.

### Motion Trajectory Annotataion Prepare
You can use our preprocessed annotation files or choose to process your own motion trajectory annotation files using [Co-Track](https://github.com/facebookresearch/co-tracker?tab=readme-ov-file#installation-instructions).


If you choose to generate motion trajectory annotations yourself, you need to follow the processing steps outlined in [Co-Track](https://github.com/facebookresearch/co-tracker?tab=readme-ov-file#installation-instructions).

```Shell
cd ./utils/co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard

mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
cd ..

```
Then, modify the corresponding ```video_path```, ```ann_path```, and ```save_path``` in the ```Generate_Trajectory_for_VIPSeg.sh``` file, and run the command. The corresponding trajectory annotations will be saved as .json files in the save_path directory.

```Shell
Generate_Trajectory_for_VIPSeg.sh

```

### Trajectory visualization
You can run the following command for visualization.

```Shell
cd .utils/
python vis_trajectory.py
```


## :paintbrush: Train <!-- omit in toc -->

### 1) Semantic Embedding Extraction

```Shell
cd .utils/
python extract_semantic_point.py
```

### 2) Train DragAnything

For VIPSeg
```Shell
sh ./script/train_VIPSeg.sh
```

For YouTube VOS
```Shell
sh ./script/train_youtube_vos.sh
```

## :paintbrush: Evaluation <!-- omit in toc -->

### Evaluation for [FID](https://github.com/mseitzer/pytorch-fid)

```Shell
cd utils
sh Evaluation_FID.sh
```

### Evaluation for [Fréchet Video Distance (FVD)](https://github.com/hyenal/relate/blob/main/extras/README.md)

```Shell
cd utils/Eval_FVD
sh compute_fvd.sh
```

### Evaluation for Eval_ObjMC

```Shell
cd utils/Eval_ObjMC
python ./ObjMC.py
```



## :paintbrush: Inference for single video <!-- omit in toc -->

```Shell
python infer_PointNet.py
```

### :paintbrush: Visulization of pixel motion for the generated video <!-- omit in toc -->

```Shell
cd utils/co-tracker
python demo.py
```



## 📖BibTeX
    @misc{wu2023paradiffusion,
          title={Paragraph-to-Image Generation with Information-Enriched Diffusion Model}, 
          author={Weijia Wu, Zhuang Li, Yefei He, Mike Zheng Shou, Chunhua Shen, Lele Cheng, Yan Li, Tingting Gao, Di Zhang, Zhongyuan Wang},
          year={2023},
          eprint={2311.14284},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
## 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for the wonderful work and codebase.
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for the wonderful work and codebase.
