'''
In this script, both network splits will be coded along with the central orchestrator class
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.Autoformer_EncDec import my_Layernorm, series_decomp, series_decomp_multi
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer


class SplitOneEncoder(nn.Module):
	'''
	First split of the FEDformer Encoder
	'''
	def __init__(self, configs):
		super(SplitOneEncoder, self).__init__()

		if isinstance(configs.moving_avg, list):
			self.decomp1 = series_decomp_multi(configs.moving_avg)
		else:
			self.decomp1 = series_decomp(configs.moving_avg)
		self_attention = FourierBlock(in_channels=configs.d_model,
							out_channels=configs.d_model,
							seq_len=configs.seq_len,
							modes=configs.modes,
							mode_select_method=configs.mode_select)
		self.dropout = nn.Dropout(configs.dropout)
		self.attention1 = AutoCorrelationLayer(
				self_attention,
				configs.d_model, configs.n_heads)

	def forward(self, x, attn_mask=None):
		new_x, attn = self.attention1(
			x, x, x,
			attn_mask=attn_mask
		)
		x = x + self.dropout(new_x)
		x, _ = self.decomp1(x)
		return x, attn


class SplitTwoEncoder(nn.Module):
	'''
	Second split of the FEDformer Encoder (2 Encoder Layers)
	'''
	def __init__(self, configs):
		super(SplitTwoEncoder, self).__init__()
		d_ff = configs.d_ff or 4 * configs.d_model
		if isinstance(configs.moving_avg, list):
			self.decomp2 = series_decomp_multi(configs.moving_avg)
			self.decomp3 = series_decomp_multi(configs.moving_avg)
			self.decomp4 = series_decomp_multi(configs.moving_avg)
		else:
			self.decomp2 = series_decomp(configs.moving_avg)
			self.decomp3 = series_decomp(configs.moving_avg)
			self.decomp4 = series_decomp(configs.moving_avg)

		self.dropout = nn.Dropout(configs.dropout)
		self.activation = F.relu if configs.activation == "relu" else F.gelu
		self_attention = FourierBlock(in_channels=configs.d_model,
				             out_channels=configs.d_model,
				             seq_len=configs.seq_len,
				             modes=configs.modes,
				             mode_select_method=configs.mode_select)
		self.attention2 = AutoCorrelationLayer(
				self_attention,
				configs.d_model, configs.n_heads)
		self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1, bias=False)
		self.norm = my_Layernorm(configs.d_model)

	def forward(self, x, attn_mask=None):
		y = x
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		x, _ = self.decomp2(x + y)
		# 2nd full pass through the Encoder
		new_x, attn = self.attention2(
			x, x, x,
			attn_mask=attn_mask
		)
		x = x + self.dropout(new_x)
		x, _ = self.decomp3(x)
		y = x
		y = self.dropout(self.activation(self.conv3(y.transpose(-1, 1))))
		y = self.dropout(self.conv4(y).transpose(-1, 1))
		res, _ = self.decomp4(x + y)
		# Layer normalization
		if self.norm is not None:
			res = self.norm(res)
		return res, attn


class SplitOneDecoder(nn.Module):
	def __init__(self, configs):
		super(SplitOneDecoder, self).__init__()
		self_att = FourierBlock(in_channels=configs.d_model,
			             out_channels=configs.d_model,
			             seq_len=configs.seq_len // 2 + configs.pred_len,
			             modes=configs.modes,
			             mode_select_method=configs.mode_select)
		self.self_attention = AutoCorrelationLayer(
								self_att, configs.d_model, configs.n_heads)

		if isinstance(configs.moving_avg, list):
			self.decomp1 = series_decomp_multi(configs.moving_avg)
		else:
			self.decomp1 = series_decomp(configs.moving_avg)

		self.dropout = nn.Dropout(configs.dropout)

	def forward(self, x, x_mask=None, cross_mask=None):
		x = x + self.dropout(self.self_attention(
			x, x, x,
			attn_mask=x_mask
		)[0])
		x, trend1 = self.decomp1(x)

		return x, trend1


class SplitTwoDecoder(nn.Module):
	def __init__(self, configs):
		super(SplitTwoDecoder, self).__init__()
		d_ff = configs.d_ff or 4 * configs.d_model
		cross_att = FourierCrossAttention(in_channels=configs.d_model,
			                      out_channels=configs.d_model,
			                      seq_len_q=configs.seq_len // 2 + configs.pred_len,
			                      seq_len_kv=configs.seq_len,
			                      modes=configs.modes,
			                      mode_select_method=configs.mode_select)
		self.cross_attention = AutoCorrelationLayer(
								cross_att,
                                configs.d_model, configs.n_heads)
		self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1, bias=False)

		if isinstance(configs.moving_avg, list):
			self.decomp2 = series_decomp_multi(configs.moving_avg)
			self.decomp3 = series_decomp_multi(configs.moving_avg)
		else:
			self.decomp2 = series_decomp(configs.moving_avg)
			self.decomp3 = series_decomp(configs.moving_avg)

		self.dropout = nn.Dropout(configs.dropout)
		self.projection1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.c_out, kernel_size=3, stride=1, padding=1,
		                            padding_mode='circular', bias=False)
		self.activation = F.relu if configs.activation == "relu" else F.gelu

		self.norm = my_Layernorm(configs.d_model)
		self.projection2 = nn.Linear(configs.d_model, configs.c_out, bias=True)

	def forward(self, x, cross, trend1, x_mask=None, cross_mask=None):
		x = x + self.dropout(self.cross_attention(
			x, cross, cross,
			attn_mask=cross_mask
		)[0])

		x, trend2 = self.decomp2(x)
		y = x
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		x, trend3 = self.decomp3(x + y)

		residual_trend = trend1 + trend2 + trend3
		residual_trend = self.projection1(residual_trend.permute(0, 2, 1)).transpose(1, 2)

		if self.norm is not None:
			if isinstance(self.norm, tuple):
				x = self.norm[0](x)
			else:
				x = self.norm(x)

		if self.projection2 is not None:
			x = self.projection2(x)

		return x, residual_trend


