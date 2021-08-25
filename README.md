# FCL-Net

> This is a pytorch implementation of our FCL-Net, 2021, version 1.0.



## Introduction

​    Integrating multi-scale predictions has become a mainstream paradigm in edge detection. However, most existing methods mainly focus on learning to effectively extract and fuse multi-scale features while ignoring the deficient learning capacity at fine-level branches, limiting the overall fusion performance. In light of this, we propose a novel **Fine-scale Corrective Learning Net (FCL-Net)** that exploits semantic information from deep layers to facilitate fine-scale feature learning. FCL-Net mainly consists of a **Top-down Attentional Guiding (TAG)** and **Pixel-level Weighting (PW) module**. The TAG adapts semantic attentional cues from coarse-scale prediction into guiding the fine-scale branches by learning a top-down LSTM. The PW module treats each spatial location's importance independently, promoting the fine-level branches to detect detailed edges with high confidence. We evaluate our method on three widely used datasets, BSDS500, Multicue and BIPED. Our approach significantly outperforms the baseline and achieves competitive ODS F-measure of 0.826 on BSDS500 benchmark.



## Performance

>  Here gives some examples of edge detection results, comparing with existing methods in Figure (a).  As shown in Figure (b), our method greatly improves fine-scale feature learning and detects more detailed edges with accurate location.

|                (a)                 |                   (b)                    |
| :--------------------------------: | :--------------------------------------: |
| ![compare](./examples/compare.png) | ![compare](./examples/stage-compare.png) |

> We report ODS and OIS for comparison with other previous impressive works. Moreover, we reproduce HED, RCF and BDCN, and report the performance compared with the original paper.

| **Method**  |  **ODS**  |  **OIS**  | ODS(original paper) | OIS(original paper) |
| :---------: | :-------: | :-------: | :-----------------: | :-----------------: |
|    *HED     |   0.790   |   0.805   |        0.788        |        0.808        |
|    *RCF     |   0.797   |   0.811   |        0.798        |        0.815        |
|    *RCF+    |   0.807   |   0.823   |        0.806        |        0.823        |
|   *RCF++    |   0.813   |   0.829   |        0.811        |        0.830        |
|    *BDCN    |   0.807   |   0.821   |        0.806        |        0.826        |
|   *BDCN+    |   0.810   |   0.829   |        0.820        |        0.838        |
|   *BDCN++   |   0.819   |   0.837   |        0.828        |        0.844        |
|   RCF-SEM   |   0.799   |   0.815   |          —          |          —          |
|  RCF-SEM+   |   0.808   |   0.826   |          —          |          —          |
|  RCF-SEM++  |   0.814   |   0.833   |          —          |          —          |
|  **Ours**   | **0.807** | **0.822** |          —          |          —          |
| **Ours-MS** | **0.816** |   0.833   |          —          |          —          |
|  **Ours+**  | **0.815** | **0.834** |          —          |          —          |
| **Ours++**  | **0.826** | **0.845** |          —          |          —          |



## How to Run Our Work

### Prerequisite

- Pytorch>=0.3.1
- Tensorboard
- AttrDict

### Train and Test the Network

1. **Clone this FCL-Net repository.**

2. **Prepare datasets**

   - to download current famous edge detection dataset, you can refer to https://github.com/MarkMoHR/Awesome-Edge-Detection-Papers to prepare data. PASCAL Context dataset is available [here](https://cs.stanford.edu/~roozbeh/pascal-context/), but needs to extract edges from segmentation masks. For convenience, [RCF](https://github.com/yun-liu/rcf) provides extracted annotations for NYUD and Pascal Context, as well as preprocessed datasets like BSDS500. 
   - data augmentation: [HED]( https://github.com/s9xie/hed)  and [BIPED]() have provide augmented dataset or code for augmentation, mainly including scaling and rotation. If you want to know more details about data augmentation, you can refer to the HED paper.
   - put the data into `\data` folder, and prepare the image list referring to `train_pair.lst`.

3. **Download ImageNet pretrained parameters** 

   - Here we use the pretrained vgg16_bn model to initialize the backbone model. To download parameters provide by Pytorch, you can refer to [link](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth). And you need to put it into `./pytorch_net/models/`; 
   - BDCN also provides the source code to convert a VGG16-caffe model into pytorch version. If you are interested in this, you can refer to [BDCN](https://github.com/pkuCactus/BDCN).

4. **Training**

   - configure the parameters for training in `./pytorch_net/config`, we provide configuration files for our work. We also provide the configuration files for our reproduced HED, RCF and BDCN by pytorch.

     ```python
     standard_HED.yaml  # for HED
     standard_RCF.yaml  # for RCF
     standard_BDCN.yaml  # for BDCN
     standard_BAN.yaml  # for BAN, now we only build the network according to the paper, the training code will be added in the future.
     standard_FCL.yaml  # for our work, FCL-Net
     ```

     > This reproduced BDCN has an ODS of 0.809, which is a little bit lower than original source code. 

   - submit your task;

     ```shell
     FCL_submit.sh
     ```

5. **Evaluation**

   - To evaluate the model, please refer to Testing HED part in https://github.com/s9xie/hed; 
   - Note that you need to use Piotr's Structured Forest matlab toolbox available here https://github.com/pdollar/edges, and remember to do NMS first if needed and put files  in `./matlab_code`;
   - we provide an example here `./evaluation/eval_epoch_fcl.m` of Matlab.

6. **To draw P-R curves and compare with other works**

   - you can refer to https://github.com/yun-liu/plot-edge-pr-curves for details;

7. **Our pretrained models**.

   |     Model     |                            Link                             |  ODS  |
   | :-----------: | :---------------------------------------------------------: | :---: |
   |      HED      | [baiduyun](https://pan.baidu.com/s/1-ECU0zDQEwBs6pmq7DKp-A) | 0.790 |
   |      RCF      | [baiduyun](https://pan.baidu.com/s/1IQnGh7psQk2gOZhK1_lMwg) | 0.807 |
   |     BDCN      | [baiduyun](https://pan.baidu.com/s/1bzAeBQ_9316uic044szdnQ) | 0.809 |
   | BDCN-official | [baiduyun](https://pan.baidu.com/s/1t7Y5_cgSf2tn8B5zsyPdmA) | 0.810 |
   |      FCL      | [baiduyun](https://pan.baidu.com/s/10J2k6HZAGAYNDo2S9IGWrQ) | 0.815 |
   
   > Password for baiduyun: `repr` ;



## Our Results for Comparison

- We reveal the evaluation results after NMS of our method and our reproduced version of RCF, BDCN as well as the baseline for comparison with other work in `\evaluation\eval`;

> Note that our reproduce version of BDCN employs the original code released by the authors here https://github.com/pkuCactus/BDCN. However, we didn't reach the performance caused by subtle difference in data augmentation as referred to the author He. If you want to know more details about BDCN, you can refer to [project](https://github.com/pkuCactus/BDCN). 



## Illustration and Discussion

- We have searched for the optimal setting for the loss weights of side outputs and final output, and the experiment results are listed below for reference. 

| Weights  Setting [dsn1, ..., dsn5, dsn6, final edge map] |    ODS    |    OIS    |
| :------------------------------------------------------: | :-------: | :-------: |
|            0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0             | **0.815** | **0.834** |
|            0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0             |   0.813   |   0.831   |
|            1.0, 0.8, 0.6, 0.4, 0.2, 1.0, 1.0             |   0.814   |   0.831   |
|            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0             |   0.813   |   0.832   |



## Acknowledgement

[1]. Our implementation is based on this project by [chongruo](https://github.com/chongruo/pytorch-HED), and we also refer to wonderful projects of [Liu](https://github.com/yun-liu/rcf) and [He](https://github.com/pkuCactus/BDCN). Thank you for their wonderful works and all contributors;

[2]. When doing experiments, we also emailed [Liu](https://github.com/yun-liu/rcf) and [He](https://github.com/pkuCactus/BDCN). Thanks very much for their kind responses and helpful advice. 

