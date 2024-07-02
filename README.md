# DragAnything

### <div align="center"> DragAnything: Motion Control for Anything using Entity Representation <div> 

<div align="center">
  <a href="https://weijiawu.github.io/draganything_page/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2403.07420/"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
</div>

<p align="center">
<img src="./assets/1709656085862.jpg" width="800px"/>  
<br>
</p>


## :notes: **Updates**

<!--- [ ] Mar. 13, 2024. Release the train code in **three month**.-->
- [x] July. 1, 2024. DragAnything has been accepted by ECCV 2024!
- [x] Mar. 24, 2024. Support interactive demo with gradio.
- [x] Mar. 13, 2024. Release the inference code.
- [x] Mar. 12, 2024. Rep initialization.


---

## 🐱 Abstract
We introduce DragAnything, which utilizes an entity representation to achieve motion control for any object in controllable video generation. Comparison to existing motion control methods, DragAnything offers several advantages. Firstly, trajectory-based is more user-friendly for interaction, when acquiring other guidance signals (\eg{} masks, depth maps) is labor-intensive. Users only need to draw a line~(trajectory) during interaction. Secondly, our entity representation serves as an open-domain embedding capable of representing any object, enabling the control of motion for diverse entities, including background. Lastly, our entity representation allows simultaneous and distinct motion control for multiple objects. Extensive experiments demonstrate that our DragAnything achieves state-of-the-art performance for FVD, FID, and User Study, particularly in terms of object motion control, where our method surpasses the previous state of the art (DragNUWA) by 26% in human voting.

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
<tr>
  <td style="text-align:center;"><b>Ours</b></td>
  <td><img src="./assets/1709662077471.jpg" width="177" height="100"></td>
  <td><img src="./assets/image65111.gif" width="177" height="100"></td>              
  <td><img src="./assets/image6611.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;"><b>DragNUWA</b></td>
  <td><img src="./assets/1709662293661.jpg" width="177" height="100"></td>
  <td><img src="./assets/image77.gif" width="177" height="100"></td>              
  <td><img src="./assets/image76.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;"><b>Ours</b></td>
  <td><img src="./assets/1709662429867.jpg" width="177" height="100"></td>
  <td><img src="./assets/image75.gif" width="177" height="100"></td>              
  <td><img src="./assets/image74.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;"><b>DragNUWA</b></td>
  <td><img src="./assets/1709662596207.jpg" width="177" height="100"></td>
  <td><img src="./assets/image84.gif" width="177" height="100"></td>              
  <td><img src="./assets/image85.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;"><b>Ours</b></td>
  <td><img src="./assets/1709662724643.jpg" width="177" height="100"></td>
  <td><img src="./assets/image87.gif" width="177" height="100"></td>              
  <td><img src="./assets/image88.gif" width="177" height="100"></td>
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
<table class="center">
<tr>
  <td style="text-align:center;"><b>Drag point with SAM</b></td>
  <td style="text-align:center;"><b>2D Gaussian</b></td>
    <td style="text-align:center;"><b>Generated Video</b></td>
      <td style="text-align:center;"><b>Visualization for Pixel Motion</b></td>
</tr>

<tr>
  <td><img src="./assets/1709663429471.jpg" width="177" height="100"></td>
  <td><img src="./assets/image265.gif" width="177" height="100"></td>
  <td><img src="./assets/image265 (1).gif" width="177" height="100"></td>              
  <td><img src="./assets/image268.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709663831581.jpg" width="177" height="100"></td>
  <td><img src="./assets/image274.gif" width="177" height="100"></td>
  <td><img src="./assets/image274 (1).gif" width="177" height="100"></td>              
  <td><img src="./assets/image276.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4"><b>(a) Motion Control for Foreground</b></td>
</tr>
<tr>
  <td><img src="./assets/1709664593048.jpg" width="177" height="100"></td>
  <td><img src="./assets/image270.gif" width="177" height="100"></td>
  <td><img src="./assets/image270 (1).gif" width="177" height="100"></td>              
  <td><img src="./assets/image269.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709664834397.jpg" width="177" height="100"></td>
  <td><img src="./assets/image271.gif" width="177" height="100"></td>
  <td><img src="./assets/image271 (1).gif" width="177" height="100"></td>              
  <td><img src="./assets/image272.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4"><b>(b) Motion Control for Background</b></td>
</tr>
<tr>
  <td><img src="./assets/1709665073460.jpg" width="177" height="100"></td>
  <td><img src="./assets/image279.gif" width="177" height="100"></td>
  <td><img src="./assets/image278.gif" width="177" height="100"></td>              
  <td><img src="./assets/image277.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709665252573.jpg" width="177" height="100"></td>
  <td><img src="./assets/image282.gif" width="177" height="100"></td>
  <td><img src="./assets/image280.gif" width="177" height="100"></td>              
  <td><img src="./assets/image281.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4"><b>(c) Simultaneous Motion Control for Foreground  and Background
</b></td>
</tr>
<tr>
  <td><img src="./assets/1709665505339.jpg" width="177" height="100"></td>
  <td><img src="./assets/image283.gif" width="177" height="100"></td>
  <td><img src="./assets/image283 (1).gif" width="177" height="100"></td>              
  <td><img src="./assets/image285.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709666205795.jpg" width="177" height="100"></td>
  <td><img src="./assets/image286.gif" width="177" height="100"></td>
  <td><img src="./assets/image288.gif" width="177" height="100"></td>              
  <td><img src="./assets/image287.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709666401284.jpg" width="177" height="100"></td>
  <td><img src="./assets/image289.gif" width="177" height="100"></td>
  <td><img src="./assets/image290.gif" width="177" height="100"></td>              
  <td><img src="./assets/image291.gif" width="177" height="100"></td>
</tr>
<tr>
  <td><img src="./assets/1709666772216.jpg" width="177" height="100"></td>
  <td><img src="./assets/image294.gif" width="177" height="100"></td>
  <td><img src="./assets/image293.gif" width="177" height="100"></td>              
  <td><img src="./assets/image292.gif" width="177" height="100"></td>
</tr>
<tr>
  <td style="text-align:center;" colspan="4"><b>(d) Motion Control for Camera Motion
</b></td>
</tr>

</table>

## 🔧 Dependencies and Dataset Prepare

### Dependencies
- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)

```Shell
git clone https://github.com/Showlab/DragAnything.git
cd DragAnything

conda create -n DragAnything python=3.8
conda activate DragAnything
pip install -r requirements.txt
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

### Pretrained Model Preparation

We adopt the [ChilloutMix](https://civitai.com/models/6424/chilloutmix) as pretrained model for extraction of entity representation, please download the diffusers version:

```bash 
mkdir -p utils/pretrained_models
cd utils/pretrained_models

# Diffusers-version ChilloutMix to utils/pretrained_models
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
```

And you can download our pretrained model for the controlnet:
```bash 
mkdir -p model_out/DragAnything
cd model_out/DragAnything

# Diffusers-version DragAnything to model_out/DragAnything
git-lfs clone https://huggingface.co/weijiawu/DragAnything
```



## :paintbrush: Train(Awaiting release) <!-- omit in toc -->

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
python demo.py
```

or run the interactive inference with gradio (install the ```gradio==3.50.2```).
```Shell
cd ./script
```
download the weight of ```sam_vit_h_4b8939.pth``` from [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

```Shell
python gradio_run.py
```


### :paintbrush: Visulization of pixel motion for the generated video <!-- omit in toc -->

```Shell
cd utils/co-tracker
python demo.py
```



## 📖BibTeX
    @misc{wu2024draganything,
          title={DragAnything: Motion Control for Anything using Entity Representation}, 
          author={Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, Di Zhang},
          year={2024},
          eprint={2403.07420},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    
    
## 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for the wonderful work and codebase.
- Thanks to [svd-temporal-controlnet](https://github.com/CiaraStrawberry/svd-temporal-controlnet) for the wonderful work and codebase.
- Thanks to chaojie for building [ComfyUI-DragAnything](https://github.com/chaojie/ComfyUI-DragAnything).


