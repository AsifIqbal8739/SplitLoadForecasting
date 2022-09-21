''' Here we given the implementation of Grid Station Class which will be responsible for
- Ask its clients to perform forward pass on their dataset batch (done @ GS implementation wise)
- combines the client outputs and returns them to the main model caller
- receives the final output of split 2 for its data and returns the computed loss
- receive the gradients of upper split from the main model and performs back-propagation on its model split
- other main functions that are needed
	- set optimizer for its split
	- set criterion for loss compute
	- get data from data_proivider's custom made class
	- zero_grad function
	- step function
	- train and eval mode setting functions
	- forward function
	- backward function
'''
import torch
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from split_FED.module_client_server import Client, Server
from split_FED.split_model import SplitOneEncoder, SplitTwoEncoder, SplitOneDecoder, SplitTwoDecoder
import PracticeScripts.HMineModules as hmm
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider


class GridStation(torch.nn.Module):
	def __init__(self, configs, gsID):
		super().__init__()
		print(f'Initializing GridStation id: {gsID} !!!!')
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.version = configs.version
		self.mode_select = configs.mode_select
		self.modes = configs.modes
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.gsID = gsID    # Grid Station ID
		self.max_clients = configs.max_clients
		self.configs = configs

		kernel_size = configs.moving_avg
		if isinstance(kernel_size, list):
			self.decomp = series_decomp_multi(kernel_size)
		else:
			self.decomp = series_decomp(kernel_size)

		self.client = Client(SplitOneEncoder(configs), SplitOneDecoder(configs))
		# Embedding
		# The series-wise connection inherently contains the sequential information.
		# Thus, we can discard the position embedding of transformers.
		self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
		                                          configs.dropout)
		self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
		                                          configs.dropout)
		# if val or test, dataset will be loaded later, but selected clients stays fix
		self.train_dataset, train_loader = self._get_data(flag='train')
		self.val_dataset, val_loader = self._get_data(flag='val')
		self.test_dataset, test_loader = self._get_data(flag='test')
		self.total_clients = self.train_dataset.data_x.shape[1]
		# num_yielded filed of iterator keeps track of how many batches have
		self.selected_clients = torch.randperm(self.total_clients)[:self.max_clients]
		self.loader_dict = {
			'train' : train_loader,
			'val'   : val_loader,
			'test'  : test_loader
		}

	def _get_data(self, flag):   # flag - train, val or test
		data_set, data_loader = data_provider(self.configs, flag, self.gsID)
		return data_set, data_loader

	def set_optimizers(self, model_optim):
		# client opt encoder ... server opt decoder
		self.coe = model_optim[0]
		self.cod = model_optim[1]

	def _adjustLR(self, adjLR, epoch):
		adjLR(self.coe, epoch + 1, self.configs)
		adjLR(self.cod, epoch + 1, self.configs)

	def zero_grad(self):
		self.coe.zero_grad()
		self.cod.zero_grad()

	def step(self):
		self.coe.step()
		self.cod.step()

	def train(self):
		self.training = True  # to be used in MI calc setup
		self.client.train()

	def eval(self):
		self.training = False
		self.client.eval()

	def backward_GS(self, grad_to_c_enc, grad_to_c_dec_x, grad_to_c_dec_tr1):
		# execute client - back propagation
		self.client.client_backward(grad_to_c_enc, grad_to_c_dec_x, grad_to_c_dec_tr1)

	def _load_dataset_mode(self, mode='train'):
		# Getting the dataset and dataloader so it keeps track of which batch to train on
		self.loader = self.loader_dict[mode]
		self.numBatches = len(self.loader)
		self.iter_loader = iter(self.loader)
		self.batchYielded = 0
		if self.configs.rand_clients and mode == 'train':
			self.selected_clients = torch.randperm(self.total_clients)[:self.max_clients]
			print('New Clients Selected!!!')

	def forward(self, mode='train'):
		if self.numBatches > self.batchYielded:
			self.batchYielded += 1
			# getting data from specific clients and specific loader (train, test, val)
			x_enc, batch_y, x_mark_enc, x_mark_dec = self._select_clients()
			x_enc = x_enc.float().to(self.device)
			batch_y = batch_y.float().to(self.device)
			x_mark_enc = x_mark_enc.float().to(self.device)
			x_mark_dec = x_mark_dec.float().to(self.device)
			# to be used for loss computation later (has clients in last dim)
			self.batch_y = batch_y.to(self.device) # its batch size will be used later in SplitGSSP

			# decomp init
			mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
			# print(x_enc.device, mean.device)
			seasonal_init, trend_init = self.decomp(x_enc)

			# seasonal_init = seasonal_init.to(self.device)
			# trend_init = trend_init.to(self.device)
			# decoder input
			trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
			seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

			csie_clients = []; csidx_clients = []; csidtrend1_clients =[]
			for cc in range(self.max_clients):

				# enc - Need to split the Encoder network here
				enc_out = self.enc_embedding(x_enc[:, :, cc].unsqueeze(2), x_mark_enc)
				dec_out = self.dec_embedding(seasonal_init[:, :, cc].unsqueeze(2), x_mark_dec)

				# Client split forward pass (interim outputs)
				csie, csidx, csidtrend1 = self.client(enc_out, dec_out)
				csie_clients.append(csie)
				csidx_clients.append(csidx)
				csidtrend1_clients.append(csidtrend1)
				# remember to reset require_grad before sending to server side propagation
			return self.batchYielded, (trend_init, csie_clients, csidx_clients, csidtrend1_clients)

		return None, (-1, -1, -1, -1)   # when the GS has exhausted all client data

	# compute loss over your own data
	def compute_loss(self, criterion, outputs):
		batch_y = self.batch_y[:, -self.pred_len:, :].to(self.device)
		outputs = outputs[:, -self.pred_len:, :]
		loss = criterion(outputs, batch_y)
		return loss

	def _select_clients(self, clientIDs=None):
		if clientIDs is None:
			clientIDs = self.selected_clients
		# these batches have data from all clients, select a few and perform forward prop
		batch_x, batch_y, batch_x_mark, batch_y_mark = next(self.iter_loader)
		batch_x = batch_x[:, :, clientIDs]
		batch_y = batch_y[:, :, clientIDs]
		return batch_x, batch_y, batch_x_mark, batch_y_mark

	# this function will be used to run forward pass on other GS data batches
	def test_data(self, gsModule):  # gsModule from which to take the data

		# getting data from specific clients and specific loader (train, test, val)
		x_enc, batch_y, x_mark_enc, x_mark_dec = gsModule._select_clients()
		x_enc = x_enc.float().to(self.device)
		batch_y = batch_y.float().to(self.device)
		x_mark_enc = x_mark_enc.float().to(self.device)
		x_mark_dec = x_mark_dec.float().to(self.device)
		# to be used for loss computation later (has clients in last dim)
		#batch_y = batch_y.to(self.device) # its batch size will be used later in SplitGSSP

		# decomp init
		mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
		# print(x_enc.device, mean.device)
		seasonal_init, trend_init = self.decomp(x_enc)

		# seasonal_init = seasonal_init.to(self.device)
		# trend_init = trend_init.to(self.device)
		# decoder input
		trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
		seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

		csie_clients = []; csidx_clients = []; csidtrend1_clients =[]
		for cc in range(gsModule.max_clients):

			# enc - Need to split the Encoder network here
			enc_out = self.enc_embedding(x_enc[:, :, cc].unsqueeze(2), x_mark_enc)
			dec_out = self.dec_embedding(seasonal_init[:, :, cc].unsqueeze(2), x_mark_dec)

			# Client split forward pass (interim outputs)
			csie, csidx, csidtrend1 = self.client(enc_out, dec_out)
			csie_clients.append(csie)
			csidx_clients.append(csidx)
			csidtrend1_clients.append(csidtrend1)
			# remember to reset require_grad before sending to server side propagation
			# ADV. batch_y shared for storage
		return batch_y, (trend_init, csie_clients, csidx_clients, csidtrend1_clients)
