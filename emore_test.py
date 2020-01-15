import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
from tqdm import tqdm
from data.data_pipe import get_test_dataset


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp')

    return lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame, vgg2_fp_issame


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = torch.Tensor((img_ten.cpu().numpy()[:,:,::-1]).copy()).cuda()#.copy()是为了负索引

    return hfliped_imgs

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(dataLoader, length,embedding_size, backbone,issame, nrof_folds=10, tta=True):
    idx = 0
    embeddings = np.zeros([length, embedding_size])
    #prefetcher = DataPrefetcher(dataLoader)
    with torch.no_grad():
        begin = 0
        for i,(imgs,labels) in enumerate(tqdm(dataLoader)):
            imgs = imgs.permute(0, 3, 1, 2).float().cuda()
            embedding = backbone(imgs)
            if tta:
                flipImgs = hflip_batch(imgs)
                embedding += backbone(flipImgs)
            embeddings[begin:begin+imgs.shape[0]] = np.copy(l2_norm(embedding).cpu().data.numpy())
            begin = begin+imgs.shape[0]

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    return float(accuracy.mean())
def eval_emore_bmk(conf,net,bmk_name):
    test_set = get_test_dataset(conf,bmk_name)
    acc = perform_val(test_set['test_dataset_iter'],test_set['num_class'],512,net,test_set['issame'])
    result = {'acc': acc, 'bmname':bmk_name,'feat_name':'stu_fc1'}
    txt_output = '{:>25s}, {:>15s}, {}'.format(bmk_name, 'stu_fc1',
                                               'acc='+str(acc))
    print(txt_output)
    return acc

def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
