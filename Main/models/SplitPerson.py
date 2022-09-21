'''
Here we setup the main model of the SplitGSSP framework, it also has the SP side
of the model.
- This main model will be called from another file to perform a complete training epoch
- It will call GridStations for forward passes and forwards the activations to split 2 server side

- It performs forward prop on the received batch activations
- Final output is returned to the Main model for loss computation
- Back propagation
'''
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from split_FED.GridStation import GridStation
from split_FED.ServiceProvider import ServiceProvider
from split_FED.module_client_server import Server
from split_FED.split_model import SplitTwoEncoder, SplitTwoDecoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
	def __init__(self, configs):
		super().__init__()
		self.version = configs.version
		self.mode_select = configs.mode_select
		self.modes = configs.modes
		self.seq_len = configs.seq_len
		self.label_len = configs.label_len
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention
		self.configs = configs
		self.nGS = configs.num_neigh

		GSM = [GridStation(self.configs, id).to(device) for id in range(self.nGS)]
		SPM = [ServiceProvider(self.configs, id).to(device) for id in range(self.nGS)]
		#self.GSNumBatches = [self.GridStationModules[id].numBatches for id in range(self.nGS) ]
		# need to save the modules in a ModuleList otherwise the model doesn't gets saved
		self.GridStationModules = nn.ModuleList(GSM)
		self.ServiceProviderModules = nn.ModuleList(SPM)

		self._select_optimizer()
		self.criterion = nn.MSELoss()

	def _select_optimizer(self):
		self._setGS_optimizers()
		self._setSP_optimizers()

	def _setGS_optimizers(self):
		for id, gsmodels in enumerate(self.GridStationModules):
			model_optim = [optim.Adam(gsmodels.client.client_encoder.parameters(),
			                          lr=self.configs.learning_rate),
			               optim.Adam(gsmodels.client.client_decoder.parameters(),
			                          lr=self.configs.learning_rate)]
			gsmodels.set_optimizers(model_optim)

	def _setSP_optimizers(self):
		for id, spmodels in enumerate(self.ServiceProviderModules):
			model_optim = [optim.Adam(spmodels.server_encoder.parameters(),
			                          lr=self.configs.learning_rate),
			               optim.Adam(spmodels.server_decoder.parameters(),
			                          lr=self.configs.learning_rate)]
			spmodels.set_optimizers(model_optim)

	# to adjust learning rates of all optimizers
	def _adjustLR(self, adjLR, epoch):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID]._adjustLR(adjLR, epoch)
			self.ServiceProviderModules[gsID]._adjustLR(adjLR, epoch)

	def train(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].train()
			self.ServiceProviderModules[gsID].train()

	def eval(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].eval()
			self.ServiceProviderModules[gsID].eval()

	def zero_grad(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].zero_grad()
			self.ServiceProviderModules[gsID].zero_grad()
	
	def step(self, gsID):
		self.GridStationModules[gsID].step()
		self.ServiceProviderModules[gsID].step()

	def backward(self, gsID):
		# execute server - back propagation
		grad_cenc, grad_cdecx, grad_cdect = self.ServiceProviderModules[gsID].server_backward()
		# execute GS - back propagation - Gradient averaging first
		grad_cenc, grad_cdecx, grad_cdect = self._avgGrads(grad_cenc, grad_cdecx, grad_cdect, gsID)
		self.GridStationModules[gsID].backward_GS(grad_cenc, grad_cdecx, grad_cdect)

	# grads needs to be averaged for each GS
	def _avgGrads(self, grad_cenc, grad_cdecx, grad_cdect, gsID):
		GSModule = self.GridStationModules[gsID]
		bs = GSModule.batch_y.shape[0]
		cenc_clients = [];
		cdecx_clients = [];
		cdect_clients = []
		for ii in range(GSModule.max_clients):
			cenc_clients.append(grad_cenc[ii * bs:(ii + 1) * bs, :, :].unsqueeze(3))
			cdecx_clients.append(grad_cdecx[ii * bs:(ii + 1) * bs, :, :].unsqueeze(3))
			cdect_clients.append(grad_cdect[ii * bs:(ii + 1) * bs, :, :].unsqueeze(3))
		# taking mean over clients samples' grads
		temp_cenc = torch.mean(torch.cat((cenc_clients), dim=3), dim=3)
		temp_cdecx = torch.mean(torch.cat((cdecx_clients), dim=3), dim=3)
		temp_cdect = torch.mean(torch.cat((cdect_clients), dim=3), dim=3)

		return temp_cenc, temp_cdecx, temp_cdect  # averaged grads

	def forward(self, epoch=0, mode='train'):
		for gsID, GSModule in enumerate(self.GridStationModules): # initialize the loaders
			GSModule._load_dataset_mode(mode=mode)
		train_loss = []
		maxBatches = len(self.GridStationModules[0].loader_dict[mode])

		for it in range(maxBatches):
			self.zero_grad()
			loss = self._batch_train(mode)
			train_loss.append(np.average(loss))
			if (it % 25) == 0:
				print("\tMode: {}, iters: {}/{},  epoch: {} | loss: {:.6f}".format(mode, it + 1, maxBatches, epoch + 1, train_loss[it]))
		return train_loss

	def _batch_train(self, mode='train'):
		losses_gs = []
		# iterate through each set GS-SP
		for gsID in range(self.nGS):
			GSModule = self.GridStationModules[gsID]
			SPModule = self.ServiceProviderModules[gsID]
			batchNum, activations = GSModule(mode)  # Get batch activations of Split 1
			if batchNum is None:
				break
			else:
				trend_init, csie, csidx, csidt1 = activations
			csie, csidx, csidt1 = self._batchify(csie, csidx, csidt1)
			# Reset the csie ... requires grad OTHERWISE grads wont propagate
			csie = csie.detach().requires_grad_()
			csidx = csidx.detach().requires_grad_()
			csidt1 = csidt1.detach().requires_grad_()

			seasonal_part, trend_part = SPModule(csie, csidx, csidt1)
			# convert the full tensors into a tensor -> batch x t x nClients
			seasonal_part, trend_part = self._reshapeGS(seasonal_part, trend_part, gsID)

			# final (Initial trend was not passed to either encoder or decoder of the client
			dec_out = trend_init + trend_part + seasonal_part

			loss = GSModule.compute_loss(self.criterion, dec_out)
			if mode == 'train':
				loss.backward(retain_graph=True)
				self.backward(gsID)
				self.step(gsID)
			losses_gs.append(loss.item())
		return losses_gs

	def _loss_GS(self, dec_out):
		#bIds = torch.cat((torch.tensor([0]), self.batchSize)).cumsum(0)
		loss = torch.zeros(size=(self.nGS,))
		for gsID, GSModule in enumerate(self.GridStationModules):
			# bsize used by each GS
			outputs = dec_out[gsID]
			loss[gsID] = GSModule.compute_loss(self.criterion, outputs)
		loss = torch.mean(loss)
		return loss

	def _batchify(self, csie, csidx, csidt1):
		csie = self._batchify_helper(csie)
		csidx = self._batchify_helper(csidx)
		csidt1 = self._batchify_helper(csidt1)
		return csie, csidx, csidt1

	def _batchify_helper(self, data):
		temp = []
		for bs in range(self.configs.max_clients):
			temp.append(data[bs])
		temp = torch.cat((temp), 0)
		return temp

	def _reshapeGS(self, season, trend, gsID):
		GSModule = self.GridStationModules[gsID]
		bs = GSModule.batch_y.shape[0]
		s_clients = []; t_clients = []
		for ii in range(GSModule.max_clients):
			s_clients.append(season[ii*bs:(ii+1)*bs, :, 0].unsqueeze(2))
			t_clients.append(trend[ii * bs:(ii + 1) * bs, :, 0].unsqueeze(2))
		s_clients = torch.cat((s_clients), dim=2)
		t_clients = torch.cat((t_clients), dim=2)

		return s_clients, t_clients

	def _combineTrend(self, tre_in, trend_part, seasonal_part):
		dec_out = []
		for gsID in range(self.nGS):
			dec_out.append(tre_in[gsID] + trend_part[gsID] + seasonal_part[gsID])
		return dec_out

	def testing(self, mode='test'):
		# get test scores for each test data on their respective trained models (split 1s)
		for gsID, GSModule in enumerate(self.GridStationModules): # initialize the loaders
			GSModule._load_dataset_mode(mode=mode)
		maxBatches = len(self.GridStationModules[0].loader_dict[mode]) # for test only
		preds = []; trues = []  # for batches
		for it in range(maxBatches):
			tre_in = []; csie = [];	csidx = [];	csidt1 = []
			self.batchSize = torch.zeros((self.nGS))
			pred = [];	true = []  # for GSs
			# iterate through each GS for activations of final split 1 layer
			for gsID in range(self.nGS):
				GSModule = self.GridStationModules[gsID]
				SPModule = self.ServiceProviderModules[gsID]
				batchNum, activations = GSModule(mode)  # Get batch activations of Split 1
				if batchNum is None:
					break
				else:
					trend_init, csie, csidx, csidt1 = activations
				csie, csidx, csidt1 = self._batchify(csie, csidx, csidt1)
				# Reset the csie ... requires grad OTHERWISE grads wont propagate
				csie = csie.detach().requires_grad_()
				csidx = csidx.detach().requires_grad_()
				csidt1 = csidt1.detach().requires_grad_()

				seasonal_part, trend_part = SPModule(csie, csidx, csidt1)
				# convert the full tensors into a tensor -> batch x t x nClients
				seasonal_part, trend_part = self._reshapeGS(seasonal_part, trend_part, gsID)

				# final (Initial trend was not passed to either encoder or decoder of the client
				dec_out = trend_init + trend_part + seasonal_part
				pred.append(dec_out) # list with nGS tensors
				true.append(GSModule.batch_y)

			preds.append(pred)   # batch x nGS
			trues.append(true)
			print(f'Mode: TEST: Running Forward Pass on batch: {it+1} out of total: {maxBatches}')
		return preds, trues

	def _getbatch_ys(self, gsID):
		true = []
		for gsID, GSModule in enumerate(self.GridStationModules):
			true.append(GSModule.batch_y)
		return true

	# Adv. testing on all Neighborhood dataset batches
	# ID_1 -> main model, ID_2 -> data to be tested
	def testing_neigh(self, ID_1, ID_2, mode='test'):
		GSM_1 = self.GridStationModules[ID_1]
		SPM_1 = self.ServiceProviderModules[ID_1]
		# the above are the full pesonalized model
		GSM_2 = self.GridStationModules[ID_2] # the data to be used for prediction
		# initialize the loaders
		GSM_2._load_dataset_mode(mode=mode)
		maxBatches = len(GSM_2.loader_dict[mode])  # for test only
		preds = [];
		trues = []  # to store predictions and GTs from all GS
		for it in range(maxBatches):
			# iterate through each GS for activations of final split 1 layer
			batch_y, activations = GSM_1.test_data(GSM_2)  # Get batch activations of Split 1
			# batch_y -> 32 x 144 x maxC
			tre_in = activations[0];			csie = activations[1]
			csidx = activations[2];			csidt1 = activations[3]
			# csie->list(maxC) -> 32 x 96 x 512; tre_in -> size(batch_y)
			csie, csidx, csidt1 = self._batchify_neigh(csie, csidx, csidt1)
			# csie -> (32xmaxC) x 96 x 512
			# forwarding to server side # output seas -> (32xmaxC) x 144 x 1
			seasonal_part, trend_part = SPM_1(csie, csidx, csidt1)
			# convert the full tensors into a list of nGS with batch x t x nClients
			seasonal_part, trend_part = self._reshapeGS_neigh(seasonal_part, trend_part, GSM_2.max_clients)
			# final (Initial trend was not passed to either encoder or decoder of the client
			# seas -> 32 x 144 x maxC
			dec_out = tre_in + trend_part + seasonal_part
			preds.append(dec_out[:, -self.configs.pred_len:, 0:])
			trues.append(batch_y[:, -self.configs.pred_len:, 0:])
			if (it+1)%50 == 0:
				print(f'Mode: TEST: Running Forward Pass on batch: {it + 1} out of total: {maxBatches}')
		return preds, trues

	def _batchify_neigh(self, csie, csidx, csidt1):
		csie = torch.cat(csie, dim=0)
		csidx = torch.cat(csidx, dim=0)
		csidt1 = torch.cat(csidt1, dim=0)
		return csie, csidx, csidt1

	def _reshapeGS_neigh(self, season, trend, maxC):
		s_clients = [];
		t_clients = []
		bs = self.configs.batch_size
		for ii in range(maxC):
			s_clients.append(season[ii * bs:(ii + 1) * bs, :, 0].unsqueeze(2))
			t_clients.append(trend[ii * bs:(ii + 1) * bs, :, 0].unsqueeze(2))
		s_clients = torch.cat((s_clients), dim=2)
		t_clients = torch.cat((t_clients), dim=2)

		return s_clients, t_clients
