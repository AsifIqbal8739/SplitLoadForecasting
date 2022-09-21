# helper functions for H-MINE test script
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DummyNetwork(nn.Module):
	def __init__(self, inDim=5, hidDim = 100, outDim=5):
		super().__init__()
		self.L1 = nn.Linear(inDim, hidDim)
		self.L2 = nn.Linear(hidDim, outDim)

	def forward(self, x):
		x = F.relu(self.L1(x))
		x = self.L2(x)
		return x


class ConcatLayer(nn.Module):
	def __init__(self, dim=1):
		super().__init__()
		self.dim = dim

	def forward(self, x, y):
		return torch.cat((x, y), self.dim)


class HMineModel(nn.Module):
	def __init__(self, d1=5, d2=5, bmDim=30, outDim=1, bSize=32):
		super().__init__()
		self.bSize = bSize  # batch size
		self.d1 = d1        # dim of X,
		self.d2 = d2        # dim of Y ~ f(X)
		self.bmDim = bmDim  # block model output dimension
		self.outDim = outDim
		self.ccLayer = ConcatLayer(dim=1)
		self.BlockModel = nn.Sequential(
			nn.Linear(d1 + d2, 100, bias=True),
			nn.ReLU(),
			nn.Linear(100, bmDim)
		)
		self.MixModel = nn.Sequential(
			nn.Linear(bmDim * self.bSize, 100),
			nn.ReLU(),
			nn.Linear(100, self.outDim)
		)

	def forward(self, x, y):
		z = self.ccLayer(x, y)
		h = self.BlockModel(z)
		hr = h.reshape(-1)
		out = self.MixModel(hr)

		return out


class Mine(nn.Module):
	def __init__(self, d1=5, d2=5, hidden_size=100):
		super().__init__()
		self.ccLayer = ConcatLayer(dim=1)
		self.d1 = d1
		self.d2 = d2
		self.fc1 = nn.Linear(d1+d2, hidden_size)
		self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
		self.fc3 = nn.Linear(int(hidden_size/2), 1)
		nn.init.normal_(self.fc1.weight, std=0.02)
		nn.init.constant_(self.fc1.bias, 0)
		nn.init.normal_(self.fc2.weight, std=0.02)
		nn.init.constant_(self.fc2.bias, 0)
		nn.init.normal_(self.fc3.weight, std=0.02)
		nn.init.constant_(self.fc3.bias, 0)

	def reset_parameters(self):
		nn.init.normal_(self.fc1.weight, std=0.02)
		nn.init.constant_(self.fc1.bias, 0)
		nn.init.normal_(self.fc2.weight, std=0.02)
		nn.init.constant_(self.fc2.bias, 0)
		nn.init.normal_(self.fc3.weight, std=0.02)
		nn.init.constant_(self.fc3.bias, 0)

	def forward(self, input):
		#output = self.ccLayer(x, y)
		output = F.elu(self.fc1(input))
		output = F.elu(self.fc2(output))
		output = self.fc3(output)
		return output

def sample_batch_T(data, batch_size=100, sample_mode='joint'):
	X, Y = data
	d1, d2 = (X.shape[-1], Y.shape[-1])
	Xvec = X.reshape(-1, d1)
	Yvec = Y.reshape(-1, d2)
	nSamp = Xvec.shape[0]
	if sample_mode == 'joint':
		index = np.random.choice(range(nSamp), size=batch_size, replace=False)
		batch = torch.cat((Xvec[index], Yvec[index]), dim=1)
	else:
		joint_index = np.random.choice(range(nSamp), size=batch_size, replace=False)
		margi_index = np.random.choice(range(nSamp), size=batch_size, replace=False)
		batch = torch.cat((Xvec[joint_index], Yvec[margi_index]), dim=1)
	return batch

def mutual_information(joint, marginal, mine_net):
	t = mine_net(joint)
	et = torch.exp(mine_net(marginal))
	mi_lb = torch.mean(t) - torch.log(torch.mean(et))
	return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
	# batch is a tuple of (joint, marginal)
	joint, marginal = batch
	#joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
	#marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
	mi_lb, t, et = mutual_information(joint, marginal, mine_net)
	ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

	# unbiasing use moving average (remember to detach the movAvg!!)
	loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
	# use biased estimator
	#     loss = - mi_lb

	mine_net_optim.zero_grad()
	#autograd.backward(loss)
	loss.backward()
	mine_net_optim.step()
	return mi_lb, ma_et, loss

def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3)):
	# data is (x, y)
	log_freq = iter_num//5
	result = list()
	loss = list()
	ma_et = 1.
	for i in range(iter_num):
		batch = sample_batch_T(data, batch_size=batch_size) \
			, sample_batch_T(data, batch_size=batch_size, sample_mode='marginal')
		mi_lb, ma_et, ll = learn_mine(batch, mine_net, mine_net_optim, ma_et)
		result.append(mi_lb.detach().cpu().numpy())
		loss.append(ll.detach().cpu().numpy())
		#if (i + 1) % (log_freq) == 0:
		#	print(f'Iter: {i+1}, MI: {result[-1]}, Loss: {loss[-1]}')
	return result

def ma(a, window_size=100):
	return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]


# Class to perform Mutual Info Calculations and return the results
class MICalculator(nn.Module):
	def __init__(self, d1, d2, lr=1e-3, batch_size=100, iter_num=int(1e4)):
		super().__init__()
		self.d1 = d1
		self.d2 = d2
		self.lr = lr
		self.batch_size = batch_size
		self.iter_num = iter_num
		self.model = Mine(d1=self.d1, d2=self.d2).to(device)
		self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

	def forward(self, x1, x2, y, pplot=0):
		# x1 -> time series batch, x2 -> respective time encoding
		# y -> client split 1 output
		self.model.reset_parameters()
		x1d = x1.detach()   # Not to be included into the gradient graph
		x2d = x2.detach()
		yd = y.detach()
		inp = torch.cat((x1d, x2d), dim=2)
		result = train((inp, yd), mine_net=self.model, mine_net_optim=self.optim,
		               batch_size=self.batch_size, iter_num=self.iter_num)

		result_ma = ma(result, window_size=250)
		print("Model on: {}, Final MI: {:.4f}".format(next(self.model.parameters()).device, np.mean(result_ma[-100:])))
		if pplot:
			plt.plot(range(len(result_ma)), result_ma)
			plt.show()
			print('Curve displayed')

		return result_ma
