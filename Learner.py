import torch
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

plt.switch_backend('agg')
from utils import get_time, gen_plot, separate_bn_paras
from data.data_pipe import get_train_loader  # construct_msr_dataset
from model import Backbone, FocalLoss, QAMFace
from emore_test import eval_emore_bmk


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        # self.loader, self.class_num = construct_msr_dataset(conf)
        self.loader, self.class_num = get_train_loader(conf)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        if not inference:
            self.milestones = conf.milestones

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = QAMFace(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            self.focalLoss = FocalLoss()

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            self.optimizer = optim.SGD([
                {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                {'params': paras_only_bn}
            ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 1000
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 2
        else:
            self.threshold = conf.threshold

        # 多GPU训练
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(conf.device)
        self.head = torch.nn.DataParallel(self.head)
        self.head = self.head.to(conf.device)

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
                                     ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        print('resume model from ' + fixed_str)
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold=0, roc_curve_tensor=0):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            # manually decay lr
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = (imgs[:, (2, 1, 0)].to(conf.device) * 255)  # RGB
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)

                loss = self.focalLoss(thetas, labels)
                loss.backward()
                running_loss += loss.item() / conf.batch_size
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    self.model.eval()
                    for bmk in ['agedb_30', 'lfw', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'vgg2_fp']:
                        acc = eval_emore_bmk(conf, self.model, bmk)
                        self.board_val(bmk, acc)

                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, acc)

                self.step += 1

        self.save_state(conf, acc, to_save_folder=True, extra='final')

    def myValidation(self, conf):
        self.model.eval()

        for bmk in ['agedb_30', 'lfw', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'vgg2_fp']:
            eval_emore_bmk(conf, self.model, bmk)

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)
