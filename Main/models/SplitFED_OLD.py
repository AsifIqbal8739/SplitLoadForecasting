import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np
from split_FED.module_client_server import Client, Server
from split_FED.split_model import SplitOneEncoder, SplitTwoEncoder, SplitOneDecoder, SplitTwoDecoder



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
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        self.client = Client(SplitOneEncoder(configs), SplitOneDecoder(configs))
        self.server = Server(SplitTwoEncoder(configs), SplitTwoDecoder(configs))

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len // 2))
        dec_modes = int(min(configs.modes, (configs.seq_len // 2 + configs.pred_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

    def set_optimizers(self, model_optim):
        # client opt encoder ... server opt decoder
        self.coe = model_optim[0]
        self.cod = model_optim[1]
        self.soe = model_optim[2]
        self.sod = model_optim[3]

    def forward_client(self, inputs_encoder, inputs_decoder):
        # execute client - feed forward network
        to_server_encoder, to_server_decoder_x, to_server_decoder_trend1 = self.client(inputs_encoder, inputs_decoder)
        # execute server - feed forward network
        outputs_x, output_res_trend = self.server(to_server_encoder, to_server_decoder_x, to_server_decoder_trend1)

        return outputs_x, output_res_trend

    def backward(self):
        # execute server - back propagation
        grad_to_c_encoder, grad_to_c_decoder_x, grad_to_c_decoder_trend1 = self.server.server_backward()
        # execute client - back propagation
        self.client.client_backward(grad_to_c_encoder, grad_to_c_decoder_x, grad_to_c_decoder_trend1)

    def zero_grad(self):
        self.coe.zero_grad()
        self.cod.zero_grad()
        self.soe.zero_grad()
        self.sod.zero_grad()

    def step(self):
        self.coe.step()
        self.cod.step()
        self.soe.step()
        self.sod.step()

    def train(self):
        self.client.train()
        self.server.train()

    def eval(self):
        self.client.eval()
        self.server.eval()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc - Need to split the Encoder network here
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        # Client split forward pass
        csie, csidx, csidtrend1 = self.client(enc_out, dec_out)
        # Server split forward pass
        seasonal_part, trend_part = self.server(csie, csidx, csidtrend1)

        # final (Initial trend was not passed to either encoder or decoder of the client
        dec_out = trend_init + trend_part + seasonal_part

        #if self.output_attention:
            #return dec_out[:, -self.pred_len:, :], attns
        #else:
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        version = 'Fourier'
        #version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = False
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model0 = Model(configs)
    #model1 = SplitNN(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model0.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out0 = model0.forward(enc, enc_mark, dec, dec_mark)
    #out1 = model1.forward(enc, enc_mark, dec, dec_mark)
    #print(np.linalg.norm(out0.detach().numpy()-out1.detach().numpy()))
