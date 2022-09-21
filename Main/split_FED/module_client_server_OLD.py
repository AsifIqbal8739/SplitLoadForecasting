import torch
from split_FED.split_model import SplitOneEncoder, SplitTwoEncoder, SplitOneDecoder, SplitTwoDecoder


class Client(torch.nn.Module):
    def __init__(self, client_encoder, client_decoder):
        super().__init__()

        self.client_encoder = client_encoder
        self.client_decoder = client_decoder

        self.client_side_interm_encoder = None
        self.client_side_interm_decoder_x = None
        self.client_side_interm_decoder_trend1 = None

        self.grad_from_server_encoder = None
        self.grad_from_server_decoder_x = None
        self.grad_from_server_decoder_trend1 = None

    def forward(self, inputs_encoder, input_decoder):

        self.client_side_interm_encoder = self.client_encoder(inputs_encoder)[0]
        # send intermidiate tensor to the server
        client_side_interm_encoder = self.client_side_interm_encoder.detach()\
            .requires_grad_()

        self.client_side_interm_decoder_x, self.client_side_interm_decoder_trend1 = self.client_decoder(input_decoder)
        client_side_interm_decoder_x = self.client_side_interm_decoder_x.detach()\
            .requires_grad_()
        client_side_interm_decoder_trend1 = self.client_side_interm_decoder_trend1.detach() \
            .requires_grad_()
        return client_side_interm_encoder, client_side_interm_decoder_x, client_side_interm_decoder_trend1

    def client_backward(self, grad_from_server_encoder, grad_from_server_decoder_x,
                        grad_from_server_decoder_trend1):
        self.grad_from_server_encoder = grad_from_server_encoder
        self.client_side_interm_encoder.backward(grad_from_server_encoder)

        self.grad_from_server_decoder_x = grad_from_server_decoder_x
        self.client_side_interm_decoder_x.backward(grad_from_server_decoder_x)

        self.grad_from_server_decoder_trend1 = grad_from_server_decoder_trend1
        self.client_side_interm_decoder_trend1.backward(grad_from_server_decoder_trend1)

    def train(self):
        self.client_encoder.train()
        self.client_decoder.train()

    def eval(self):
        self.client_encoder.eval()
        self.client_decoder.eval()


class Server(torch.nn.Module):
    def __init__(self, server_encoder, server_decoder):
        super().__init__()
        self.server_encoder = server_encoder
        self.server_decoder = server_decoder

        self.interm_to_server_encoder = None
        self.grad_to_client_encoder = None

        self.interm_to_server_decoder_x = None
        self.grad_to_client_decoder_x = None

        self.interm_to_server_decoder_trend1 = None
        self.grad_to_client_decoder_trend1 = None

    def forward(self, interm_to_server_encoder, interm_to_server_decoder_x,
                interm_to_server_decoder_trend1):
        self.interm_to_server_encoder = interm_to_server_encoder
        self.interm_to_server_decoder_x = interm_to_server_decoder_x
        self.interm_to_server_decoder_trend1 = interm_to_server_decoder_trend1

        outputs_encoder, _ = self.server_encoder(interm_to_server_encoder)

        output_x, output_res_trend = self.server_decoder(interm_to_server_decoder_x, outputs_encoder,
                                                         interm_to_server_decoder_trend1)

        return output_x, output_res_trend

    def server_backward(self):
        self.grad_to_client_encoder = self.interm_to_server_encoder.grad.clone()
        self.grad_to_client_decoder_x = self.interm_to_server_decoder_x.grad.clone()
        self.grad_to_client_decoder_trend1 = self.interm_to_server_decoder_trend1.grad.clone()

        return self.grad_to_client_encoder, self.grad_to_client_decoder_x, self.grad_to_client_decoder_trend1

    def train(self):
        self.server_encoder.train()
        self.server_decoder.train()

    def eval(self):
        self.server_encoder.eval()
        self.server_decoder.eval()