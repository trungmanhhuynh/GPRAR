import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from base_setting import BaseSetting


class Reconstruction(BaseSetting):

    def __init__(self, argv=None):
        super().__init__(argv)

        # define loss
        self.lossReg = nn.CrossEntropyLoss()
        self.lossRec = nn.MSELoss()

        # log
        self.loss_res = {'loss_rec': [], 'loss_reg': [], 'loss': [], 'ade': []}

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.rec_result = []
                    self.test()
            self.io.print_log('Save log file')
            outfile = os.path.join(self.arg.work_dir, "loss.json")
            with open(outfile, 'w') as f:
                json.dump(self.loss_res, f)

        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.rec_result = []
            self.result = dict()
            self.io.print_log('Evaluation Start:')
            self.test()

            # save the output of model
            if self.arg.save_result:
                self.io.save_pkl(self.rec_result, 'test_result.pkl')

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for noisy_data, data, label, video_name, image_name, bbox in loader:
            # get data
            noisy_data = noisy_data.float().to(self.dev)
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output_reg, output_rec = self.model(noisy_data)
            lossReg = self.lossReg(output_reg, label)
            lossRec = self.lossRec(output_rec, data)
            loss = lossRec + lossReg

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lossReg'] = lossReg.data.item()
            self.iter_info['lossRec'] = lossRec.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value, loss_reg_value, loss_rec_value = [], [], []
        result_frag = []
        label_frag = []
        meanADE = []

        for noisy_data, gt_data, label, video_name, image_name, bbox in loader:

            # get data
            noisy_data = noisy_data.float().to(self.dev)
            gt_data = gt_data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output_reg, output_rec = self.model(noisy_data)

            result_frag.append(output_reg.data.cpu().numpy())

            for i in range(noisy_data.shape[0]):
                rec_result_frag = {'in_pose': noisy_data[i].cpu().numpy(), 'out_pose': output_rec[i].cpu().numpy(),
                                   'gt_pose': gt_data[i].cpu().numpy(), "bbox": bbox[i].cpu().numpy(),
                                   'video_name': video_name[i], 'image_name': image_name[i]}
                self.rec_result.append(rec_result_frag)

            # get loss
            if evaluation:
                lossReg = self.lossReg(output_reg, label)
                lossRec = self.lossRec(output_rec, gt_data)
                loss = lossReg + lossRec
                loss_value.append(loss.item())
                loss_rec_value.append(lossRec.item())
                loss_reg_value.append(lossReg.item())
                label_frag.append(label.data.cpu().numpy())

                ade = self.cal_ade(output_rec.data.cpu().numpy(), gt_data.data.cpu().numpy(), bbox.data.cpu().numpy())
                meanADE.append(ade.item())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss_rec'] = np.mean(loss_rec_value)
            self.epoch_info['mean_loss_reg'] = np.mean(loss_reg_value)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['ade'] = np.mean(meanADE)
            self.show_epoch_info()
            # save loss
            self.loss_res['loss_rec'].append(self.epoch_info['mean_loss_rec'])
            self.loss_res['loss_reg'].append(self.epoch_info['mean_loss_reg'])
            self.loss_res['loss'].append(self.epoch_info['mean_loss'])
            self.loss_res['ade'].append(self.epoch_info['ade'])

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def cal_ade(self, pose_res, pose_gt, bbox):
        """
            Args:

            Shape:
                bbox: # (N, 4, T, V, 1)
                pose_res:  (N, C, T, V , M)
                pose_gt:  (N, C, T, V , M)

            Returns:
        """
        # input shape ~ (N, C, T, V , M)
        N, C, T, V, M = pose_res.shape

        # we have to de-normalize pose data two times. First, use bounding box's width and height
        # and second use frame's width and height.

        pose_res[:, 0] = ((pose_res[:, 0] + 0.5) * (bbox[:, 2] - bbox[:, 0]) + bbox[:, 0]) * self.arg.W  # x
        pose_res[:, 1] = ((pose_res[:, 1] + 0.5) * (bbox[:, 3] - bbox[:, 1]) + bbox[:, 1]) * self.arg.H  # y
        pose_gt[:, 0] = ((pose_gt[:, 0] + 0.5) * (bbox[:, 2] - bbox[:, 0]) + bbox[:, 0]) * self.arg.W  # x
        pose_gt[:, 1] = ((pose_gt[:, 1] + 0.5) * (bbox[:, 3] - bbox[:, 1]) + bbox[:, 1]) * self.arg.H  # y

        temp = (pose_res[:, 0:2] - pose_gt[:, 0:2]) ** 2
        ade = np.sqrt(temp[:, 0] + temp[:, 1])
        ade = np.mean(ade)

        return ade

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = BaseSetting.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help,
                                         parents=[parent_parser],
                                         description='Parser for Reconstruction')

        parser.add_argument('--W', type=int, default=1080, help='frame width')
        parser.add_argument('--H', type=int, default=1080, help='frame height')
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')

        return parser
