import numpy as np
import torch
from prediction_settings import PredictionSettings

class Prediction(PredictionSettings):

    def __init__(self, argv=None):
        super().__init__(argv)

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
                    self.test()
                    # self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                # raise ValueError('Please appoint --weights.')
                self.io.print_log('Caution: Weight is None')
            else:
                self.io.print_log('Model:   {}.'.format(self.arg.model))
                self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for obs_loc, obs_pose, obs_gridflow, gt_location in loader:

            # get data
            obs_loc = obs_loc.float().to(self.dev)
            obs_pose = obs_pose.float().to(self.dev)
            obs_gridflow = obs_gridflow.float().to(self.dev)
            gt_location = gt_location.float().to(self.dev)

            # forward
            pred_loc = self.model((obs_loc, obs_pose, obs_gridflow))
            loss = self.loss(pred_loc, gt_location)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        # self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value, ade_value, fde_value = [], [], []
        result_frag = []

        for obs_loc, obs_pose, obs_gridflow, gt_location in loader:

            # get data
            obs_loc = obs_loc.float().to(self.dev)
            obs_pose = obs_pose.float().to(self.dev)
            obs_gridflow = obs_gridflow.float().to(self.dev)
            gt_location = gt_location.float().to(self.dev)

            # inference
            with torch.no_grad():
                pred_loc = self.model((obs_loc, obs_pose, obs_gridflow))
            result_frag.append(pred_loc.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(pred_loc, gt_location)
                loss_value.append(loss.item())

                # pred_loc = pred_loc.data.cpu().numpy()
                # gt_location = gt_location.data.cpu().numpy()
                ade, fde = self.calculate_ade_fde(pred_loc.data.cpu().numpy(), gt_location.data.cpu().numpy())
                ade_value.append(ade)
                fde_value.append(fde)

        self.result = np.concatenate(result_frag)
        if evaluation:

            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['ade'] = np.mean(ade_value)
            self.epoch_info['fde'] = np.mean(fde_value)
            self.show_epoch_info()

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

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def calculate_ade_fde(self, pred_loc, gt_loc):
        '''
            pred_loc: predicted locations (N, pred_len, 2)
            gt_loc: predicted locations (N, pred_len, 2)
        '''

        # convert to pixels
        pred_loc[:, :, 0] = (pred_loc[:, :, 0] + 0.5) * self.arg.W  # x
        pred_loc[:, :, 1] = (pred_loc[:, :, 1] + 0.5) * self.arg.H  # y
        gt_loc[:, :, 0] = (gt_loc[:, :, 0] + 0.5) * self.arg.W  # x
        gt_loc[:, :, 1] = (gt_loc[:, :, 1] + 0.5) * self.arg.H  # y

        # caculate ade/fde
        ade = np.sqrt(np.mean((pred_loc - gt_loc)**2))
        fde = np.sqrt(np.mean((pred_loc[:, -1] - gt_loc[:, -1])**2))

        return ade, fde
