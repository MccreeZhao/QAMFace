# QAMFace
Pytorch implementation for **Quadratic Additive Angular Margin Loss for Face Recognition** (under review)
## License
The code is released under the MIT License.
## News
**`2020.1.15`** We released our QAMFace loss and training codes.
## Pre-Requisites 
* Linux or macOS
* [Python 3](https://www.anaconda.com/distribution/) (for training \& validation)
* PyTorch 1.0 (for traininig \& validation, install w/ `pip install torch torchvision`)
* MXNet 1.3.1 (optional, for data processing, install w/ `pip install mxnet`)
* tensorboardX 1.4 (optional, for visualization, install w/ `pip install tensorboardX`)
* OpenCV 3 (install w/ `pip install opencv-python`)
* bcolz 1.2.0 (install w/ `pip install bcolz`)
 We used 4 NVIDIA RTX 2080Ti in parallel. More GPUs which support larger batch-size may perform better.
 
## Usage
### Data Preprocess
- Download training data from [IngishtFace Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo), we highly recommand you to use emore.
- Unzip the file. Edit the `save_path` and `rec_path` in make_extracted.py. Run this script to extract image from mx_rec data.
- Edit the `conf.data_path` in config.py with `save_path` mentioned above.
- For data augmentation, we simply apply horizental flip. If you want use more complicated process to achieve higher performance, please refer to [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/) which proved good examples.

### Model Training and Validation
- Hyper parameters such as batch-size, learning rate can be edited in train.py. 
- Hyper parameters of loss functions such as s and m can be edited in model.py
- Run train.py to train and validate the model. 
- Use tensorboard to monitor the training log : `tensorboard --logdir='./'`.

## Performance
### model

|Backbone|Head|Loss|Training Data|
  |:---:|:---:|:---:|:---:|
  |[IRSE-50](https://arxiv.org/pdf/1801.07698.pdf)|[ArcFace]( http://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)|[Focal](https://arxiv.org/pdf/1708.02002.pdf)|[emore](https://arxiv.org/pdf/1607.08221.pdf)|
  
### Setting)
- INPUT_SIZE: [112, 112]
- BATCH_SIZE: 256 (drop the last batch to ensure consistent batch_norm statistics)
- Initial LR: 0.2; 
- NUM_EPOCH: 22;
- WEIGHT_DECAY: 5e-4 (do not apply to batch_norm parameters); 
- MOMENTUM: 0.9; STAGES: [30, 60, 90]; 
- Augmentation: Horizontal Flip;
- Solver: SGD; 
- GPUs: 4 NVIDIA RTX 2080Ti in Parallel
### Performance

|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[CFP_FP](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|99.82|99.89|98.04|98.12|96.12|92.80|95.64|

## Acknowledgement 
- This repo is inspired by 
  - [InsightFace.MXNet](https://github.com/deepinsight/insightface)
  - [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch)
  - [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)
