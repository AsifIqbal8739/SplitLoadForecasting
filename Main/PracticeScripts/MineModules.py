# helper functions for MINE test script
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mine(nn.Module):
	def __init__(self, input_size=2, hidden_size=100):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)
		nn.init.normal_(self.fc1.weight, std=0.02)
		nn.init.constant_(self.fc1.bias, 0)
		nn.init.normal_(self.fc2.weight, std=0.02)
		nn.init.constant_(self.fc2.bias, 0)
		nn.init.normal_(self.fc3.weight, std=0.02)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, input):
		output = F.elu(self.fc1(input))
		output = F.elu(self.fc2(output))
		output = self.fc3(output)
		return output


def mutual_information(joint, marginal, mine_net):
	t = mine_net(joint)
	et = torch.exp(mine_net(marginal))
	mi_lb = torch.mean(t) - torch.log(torch.mean(et))
	return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
	# batch is a tuple of (joint, marginal)
	joint, marginal = batch
	joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
	marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
	mi_lb, t, et = mutual_information(joint, marginal, mine_net)
	ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

	# unbiasing use moving average
	loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
	# use biased estimator
	#     loss = - mi_lb

	mine_net_optim.zero_grad()
	autograd.backward(loss)
	mine_net_optim.step()
	return mi_lb, ma_et


def sample_batch(data, batch_size=100, sample_mode='joint'):
	if sample_mode == 'joint':
		index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
		batch = data[index]
	else:
		joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
		marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
		batch = np.concatenate([data[joint_index][:, 0].reshape(-1, 1),
		                        data[marginal_index][:, 1].reshape(-1, 1)],
		                       axis=1)
	return batch


def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
	# data is x or y
	result = list()
	ma_et = 1.
	for i in range(iter_num):
		batch = sample_batch(data, batch_size=batch_size) \
			, sample_batch(data, batch_size=batch_size, sample_mode='marginal')
		mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
		result.append(mi_lb.detach().cpu().numpy())
		if (i + 1) % (log_freq) == 0:
			print(f'Iter: {i}, MI: {result[-1]}')
	return result


def ma(a, window_size=100):
	return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]
