from config import get_config
from Learner import face_learner
import argparse
import torch
# tensorboard --logdir='./' --port=2335
import os
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(415)
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=22, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=2e-1, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=8, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat,IMDB]",
                        default='emore',
                        type=str)
    args = parser.parse_args()

    conf = get_config()

    conf.net_mode = args.net_mode  # 默认IR——SE
    conf.net_depth = args.net_depth

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode

    conf.milestones = [4, 10, 15, 20]  # learning_rate decay

    learner = face_learner(conf)
    learner.train(conf, args.epochs)
