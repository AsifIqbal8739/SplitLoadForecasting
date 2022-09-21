import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer, SplitFED
from models import SplitGSSP, SplitPerson   #
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


class Exp_SplitGSSP(Exp_Basic):
    def __init__(self, args):
        super(Exp_SplitGSSP, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'SplitFED': SplitFED,
            'SplitGSSP': SplitGSSP, # Shared SP, unique GS
            'SplitPerson': SplitPerson  # Unique SP and GS
        }
        model = model_dict[self.args.model].Model(self.args).float()
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #model = model.to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def vali(self, mode='val'):
        self.model.eval()
        with torch.no_grad():
            loss = self.model(mode=mode)
        total_loss = np.average(loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.model.train()
        train_losses = []
        vali_losses = []
        test_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_time = time.time()
            train_loss = self.model(epoch=epoch)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_losses.append(np.average(train_loss))
            vali_loss = self.vali(mode='val');  vali_losses.append(vali_loss)
            test_loss = self.vali(mode='test'); test_losses.append(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, self.model.GridStationModules[0].numBatches, train_losses[epoch], vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                torch.save(self.model.state_dict(), path+'last')
                print("Early stopping"); #break

            # update the learning rate of optimizer
            self.model._adjustLR(adjust_learning_rate, epoch)
        # Save losses for convergence comparison
        Losses = {'train': train_losses, 'vali': vali_losses, 'test': test_losses}
        np.save(path+'/Losses.npy', Losses)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        # test=0 if running after training, otherwise test=1 to load checkpoint
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device(device)))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            preds, trues = self.model.testing()

        preds = self._separteGSdata(preds)
        trues = self._separteGSdata(trues)


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Scores are computed on all the batches together, thus errors will be high
        scores = self._metricwrapper(preds, trues)
        for ii in range(self.model.nGS):
            print('GSid: {}, mae:{:.3f}, mse:{:.3f}'.format(ii, scores[ii,0], scores[ii,1]))

        with open('results.txt', 'a') as f:
            f.write(setting + "  \n")
            for ii in range(self.model.nGS):
                f.write('GSid: {}, mae:{:.3f}, mse:{:.3f}\n'.format(ii, scores[ii, 0], scores[ii, 1]))
            f.write('\n')
            f.write('\n')


        np.save(folder_path + 'metrics.npy', scores)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def _metricwrapper(self, preds, trues):
        # scores are computed as mae, mse, rmse, mape, mspe
        scores = np.zeros((self.model.nGS, 5))
        for ii in range(self.model.nGS):
            a, b, c, d, e = metric(preds[ii], trues[ii])
            temp = np.array([a, b, c, d, e])
            scores[ii, :] = temp
        return scores

    def _separteGSdata(self, preds):
        # input preds -> list(numbatches, numGS)
        # keeping only the prediction data here as well
        pro_data = []
        for nGS in range(len(preds[0])):
            temp = []
            for nB in range(len(preds)):
                tt = preds[nB][nGS][:, -self.args.pred_len:, 0:]    # only pred data
                temp.append(tt)
            temp = torch.cat(temp, dim=0).cpu().numpy()
            pro_data.append(temp)
        return pro_data # processed data

