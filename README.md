# QAMFace
Pytorch implementation for Quadratic Additive Angular Margin Loss for Face Recognition
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
 We used 4 NVIDIA RTX 2080ti in parallel. More GPUs which support larger batch-size may perform better.
 
## Usage
### Data Preprocess
- Download training data from [IngishtFace Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo), we highly recommand you to use emore.
- Unzip the file. Edit the `save_path` and `rec_path` in make_extracted.py. Run this script to extract image from mx_rec data.
```
if __name__ == '__main__':
    save_path = '../extracted'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rec_path = '/home/ying/data2/Mccree/faces_emore_new/faces_emore'

    load_mx_rec(rec_path, save_path=os.path.join(save_path, 'faces_emore_img'))

    bin_files = ['lfw', 'agedb_30', 'cfp_fp', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']

    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, (bin_files[i] + '.bin')), os.path.join(save_path, bin_files[i]))
```
- Edit the `conf.data_path` in config.py with `save_path` mentioned above.
- For data augmentation, we simply apply horizental flip. If you want use more complicated process to achieve higher performance, please refer to [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/) which proved good examples.
### Model Training and Validation
- Hyper parameters such as batch-size, learning rate can be edited in train.py.
- Run train.py to train and validate the model. 
- Use tensorboard to monitor the training log : `tensorboard --logdir='./'`.
