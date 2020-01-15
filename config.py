from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
import os
from utils import get_time


def get_config(training=True):
    conf = edict()

    conf.data_path = Path('/home/ying/data2/Mccree/Project/ArcFace/extracted/')  ## Training data path
    conf.work_path = Path('work_space/')

    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.net_depth = 50
    conf.drop_ratio = 0.4
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        # trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.data_mode = 'emore'
    conf.emore_folder = conf.data_path
    conf.batch_size = 256
    # --------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
        conf.lr = 2e-1
        conf.momentum = 0.9
        conf.pin_memory = True
        conf.ce_loss = CrossEntropyLoss()

    if not os.path.exists(conf.work_path):
        os.mkdir(conf.work_path)
    conf.model_path = conf.work_path / 'models'
    if not os.path.exists(conf.model_path):
        os.mkdir(conf.model_path)
    conf.log_path = conf.work_path / 'log' / get_time()
    if not os.path.exists(conf.log_path):
        os.makedirs(conf.log_path)
    conf.save_path = conf.work_path / 'save'
    if not os.path.exists(conf.save_path):
        os.mkdir(conf.save_path)
    return conf
