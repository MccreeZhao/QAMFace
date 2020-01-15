import argparse
import numpy as np
import cv2
import bcolz
import pickle
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import os


def load_bin(path, rootdir, image_size=[112, 112]):
    align5p = []
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    imgDir = os.path.join(rootdir, 'imgs')
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(imgDir, '{}.jpg'.format(i)), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        align5p.append(os.path.join(imgDir, '{}.jpg'.format(i)))
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)

    print(data.shape)
    np.save(os.path.join(rootdir, rootdir.split('/')[-1] + '_list'), np.array(issame_list))
    np.save(os.path.join('../processed/', 'list.npy'), np.array(issame_list))
    np.save(os.path.join(rootdir, 'list.npy'), np.array(issame_list))
    print(os.path.join(rootdir, 'align5p.npy'))
    np.save(os.path.join(rootdir, 'align5p.npy'), align5p)

    return data, issame_list


def load_mx_rec(rec_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'),
                                           'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = os.path.join(save_path, str(label))
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        cv2.imwrite(os.path.join(label_path, '{}.jpg'.format(idx)), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    save_path = '../extracted'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rec_path = '/home/ying/data2/Mccree/faces_emore_new/faces_emore'

    load_mx_rec(rec_path, save_path=os.path.join(save_path, 'faces_emore_img'))

    bin_files = ['lfw', 'agedb_30', 'cfp_fp', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']

    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, (bin_files[i] + '.bin')), os.path.join(save_path, bin_files[i]))
