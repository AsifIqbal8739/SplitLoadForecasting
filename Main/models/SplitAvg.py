# A class to implement a Split Average model from a trained SplitPerson model
# SPavg is computed using SPModules from SPlitPerson class
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models import SplitPerson
from split_FED.GridStation import GridStation
from split_FED.ServiceProvider import ServiceProvider
from split_FED.module_client_server import Server
from split_FED.split_model import SplitTwoEncoder, SplitTwoDecoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(SplitPerson.Model):
	def __init__(self, SPModel, configs):
		super().__init__(configs)
		self.configs = configs
		self.GridStationModules = SPModel.GridStationModules.to(device)
		self.ServiceProviderModules = SPModel.ServiceProviderModules.to(device)
		# Initialize average ServiceProvider Module
		self.SPAverage = ServiceProvider(self.configs, 0).to(device)
		self._initialize_SPAvg()

	def _initialize_SPAvg(self):

		# get state_dicts of all SPs
		sd0 = self.ServiceProviderModules[0].state_dict()
		sd1 = self.ServiceProviderModules[1].state_dict()
		sd2 = self.ServiceProviderModules[2].state_dict()

		sdavg = self.SPAverage.state_dict()

		# Average all parameters
		for key in sdavg:
			sdavg[key] = (sd0[key] + sd1[key] + sd2[key]) / 3.

		# Recreate model and load averaged state_dict
		self.SPAverage.load_state_dict(sdavg)

	def train(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].train()
		self.SPAverage.train()

	def eval(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].eval()
		self.SPAverage.eval()

	def zero_grad(self):
		for gsID in range(self.nGS):
			self.GridStationModules[gsID].zero_grad()
		self.SPAverage.zero_grad()

	# overload for a single line change
	# Adv. testing on all Neighborhood dataset batches
	# ID_1 -> main model, ID_2 -> data to be tested
	def testing_neigh(self, ID_1, ID_2, mode='test'):
		GSM_1 = self.GridStationModules[ID_1]
		SPM_1 = self.SPAverage
		# the above are the full pesonalized model
		GSM_2 = self.GridStationModules[ID_2] # the data to be used for prediction
		# initialize the loaders
		GSM_2._load_dataset_mode(mode=mode)
		maxBatches = len(GSM_2.loader)  # for test only
		preds = [];		trues = []  # to store predictions and GTs from all GS
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
			if (it+1)% 50 == 0:
				print(f'Mode: TEST: Running Forward Pass on batch: {it + 1} out of total: {maxBatches}')
		return preds, trues