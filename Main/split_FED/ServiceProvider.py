# Here I will implement ServiceProvider class which will be used to learn
# personalized models combined with GridStations
import torch
from split_FED.module_client_server import Server
from split_FED.split_model import SplitTwoEncoder, SplitTwoDecoder


class ServiceProvider(Server):
	def __init__(self, configs, spID):
		print(f'Initializing ServiceProvider id: {spID} !!!!')
		super().__init__(SplitTwoEncoder(configs), SplitTwoDecoder(configs))
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.spID = spID
		self.configs = configs


	def set_optimizers(self, model_optim):
		# client opt encoder ... server opt decoder
		self.soe = model_optim[0]
		self.sod = model_optim[1]

	def _adjustLR(self, adjLR, epoch):
		adjLR(self.soe, epoch + 1, self.configs)
		adjLR(self.sod, epoch + 1, self.configs)

	def zero_grad(self):
		self.soe.zero_grad()
		self.sod.zero_grad()

	def step(self):
		self.soe.step()
		self.sod.step()