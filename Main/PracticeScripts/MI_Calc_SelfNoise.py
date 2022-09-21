# Here I will extract the batches from the same dataset used for Test_1 & 2
# i.e., ETTh1.csv and compare its MI against ---

from data_provider.data_factory import data_provider
import torch
import numpy as np
import PracticeScripts.HMineModules as hmm
import warnings
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser(description='Batch Extraction')
# data loader
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='../../Datasets/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # for debugging, keep it at 0

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data, train_loader = data_provider(args, flag='train')
# MI Calculator, 4 is for hourly time resolution
MICalc = hmm.MICalculator(d1=train_data.data_x.shape[1] + 4, d2=train_data.data_x.shape[1] + 4)

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    if i%10 != 0: continue
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    temp = torch.cat((batch_x, batch_x_mark), dim=2)
    noise = torch.normal(mean=0, std=1, size=temp.shape, dtype=torch.float32, device=device)

    MI_result = MICalc(batch_x, batch_x_mark, noise, pplot=0)
    MI_result = np.array(MI_result).reshape(-1, 1)
    if i > 0:
        batch_MI_Scores = np.concatenate((batch_MI_Scores, MI_result), axis=1)
    else:
        batch_MI_Scores = MI_result
    if i%30 == 0:
        print(f'Batch: {i}:')
# Noise with input dimensions
with open('MI_Scores_wrt_Noise_InpDim.npy', 'wb') as f:
    np.save(f, batch_MI_Scores)