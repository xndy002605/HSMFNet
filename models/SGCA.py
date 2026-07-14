# =====================================================================
# SGCA: SEMANTIC-GUIDED CROSS-ATTENTION CALIBRATION MODULE
# Purpose: Calibrate fused features using high-purity semantic features
#          from the final backbone layer as an anchor.
#   - Channel Attention (CA) on semantic features: emphasize hierarchical semantics
#   - Spatial Attention (SA) on fused features: suppress background noise
#   - Cross-attention with residual connections: preserve original information
#   Core innovation: Semantic denoising and calibration, not just feature transfer.
# =====================================================================
from idlelib.query import Query

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from models.backbone.utils import LayerNorm
from models.network.conv_cross_att import ConvDecoder, LSKALight, LSKA
from models.network.mmf import BasicConv


# Attention network
class Attentions(nn.Module):
	def __init__(self, channel_size=256):
		super(Attentions, self).__init__()
		self.A1_c = PartSE(channel_size)
		self.A2_c = PartSE(channel_size)
		self.A3_c = PartSE(channel_size)
		self.A4_c = PartSE(channel_size)

	def forward(self, inputs):
		F1, F2, F3,F4 = inputs
		# Global Average Pooling to a vector
		A1_channel = self.A1_c(F1)
		A2_channel = self.A2_c(F2)
		A3_channel = self.A3_c(F3)
		A4_channel = self.A4_c(F4)

		# bottom to top
		A2_channel = (A2_channel + A1_channel) / 2
		A3_channel = (A3_channel + A2_channel) / 2
		A4_channel = (A4_channel + A3_channel) / 2

		# channel pooling
		# A1 = F1 * A1_channel
		# A2 = F2 * A2_channel
		A4 = F1 * A4_channel
		return A4


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):

		super().__init__()
		assert kernel_size % 2 == 1
		padding = kernel_size // 2

		self.sigmoid = nn.Sigmoid()

		self.conv = nn.Conv2d(
			in_channels=2,
			out_channels=1,
			kernel_size=kernel_size,
			padding=padding,
			bias=False
		)

	def forward(self, x):

		avg_pool = torch.mean(x, dim=1, keepdim=True)  # F_avg^s [B,1,H,W]

		max_pool, _ = torch.max(x, dim=1, keepdim=True)  # F_max^s [B,1,H,W]

		pooled_features = torch.cat((avg_pool, max_pool), dim=1)  # [B,2,H,W]

		spatial_attention = self.conv(pooled_features)

		spatial_attention = self.sigmoid(spatial_attention)

		return spatial_attention*x


class ChannelGate(nn.Module):
	def __init__(self, out_channels):
		super(ChannelGate, self).__init__()
		self.conv1 = nn.Conv2d(out_channels, out_channels//4, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
		self.gelu = nn.GELU()
		self.sigmoid = nn.Sigmoid()

		nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in')
		nn.init.constant_(self.conv1.bias, 0.)
		nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in')
		nn.init.constant_(self.conv2.bias, 0.)

	def forward(self, x):
		x = nn.AdaptiveAvgPool2d(output_size=1)(x)
		x = self.conv1(x)
		x = self.gelu(x)
		x = self.conv2(x)
		x = self.sigmoid(x)
		# x = F.relu(self.conv1(x), inplace=True)
		# x = torch.sigmoid(self.conv2(x))
		return x

class PartSE(nn.Module):
	def __init__(self, num_parts):
		super().__init__()
		self.gap = nn.AdaptiveAvgPool1d(1)
		self.conv1 = nn.Conv2d(num_parts, num_parts, 1)
		self.conv2 = nn.Conv2d(num_parts, num_parts, 1)
		self.ln = nn.LayerNorm([num_parts, 1, 1])
		self.activation = nn.GELU()
		self.standardize = nn.Sigmoid()

		nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in')
		nn.init.constant_(self.conv1.bias, 0.)
		nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in')
		nn.init.constant_(self.conv2.bias, 0.)

	def flops(self):
		flops = 0
		# GAP
		flops += self.Cr * self.C
		# Conv1
		flops += self.Cr * self.Cr
		# LN + Act + Sigmoid
		flops += 2 * self.Cr
		# Conv2
		flops += self.Cr * self.Cr
		return flops



	def forward(self, x):
		B, Cr, C = x.shape
		self.Cr,self.C = Cr, C
		x = self.gap(x).unsqueeze(-1)
		x = self.conv1(x)
		x = self.ln(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.standardize(x).reshape(B, Cr,1)  # (B,1,1,C/r)
		return x

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class BasicConv1(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
				 bn=True, bias=False):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
							  dilation=dilation, groups=groups, bias=bias)
		# self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU(inplace=True) if relu else None

	def forward(self, x):
		x = self.conv(x)
		# if self.bn is not None:
		#     x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class SimpleFPA(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(SimpleFPA, self).__init__()

		self.channels_cond = in_planes
		# Master branch
		self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

		# Global pooling branch
		self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

	def forward(self, x):
		# Master branch
		x_master = self.conv_master(x)

		# Global pooling branch
		x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
		x_gpb = self.conv_gpb(x_gpb)

		out = x_master + x_gpb

		return out


class PyramidFeatures(nn.Module):
	def __init__(self,dim=640,num_heads=4,att_drop=0.2):
		super(PyramidFeatures, self).__init__()

		self.mpsa_list = nn.ModuleList()
		self.a = nn.ModuleList()
		stage_scale_list = [8, 4, 2, 1]
		self.proj = nn.Linear(dim, dim)
		for i in range(3):
			self.mpsa_list.append(MutipCrossAttention(dim // stage_scale_list[i+1],dim // stage_scale_list[i],dim // num_heads,att_drop))
			self.a.append(nn.Linear(dim // stage_scale_list[i], dim // stage_scale_list[i]))


	def forward(self, inputs):

		outputs = []
		outputs.append(self.proj(inputs[-1]))
		temp = inputs[3]
		for i in reversed(range(3)):
			out = self.mpsa_list[i](inputs[i], temp)
			temp = out + inputs[i]
			outputs.append(self.a[i](temp))
		return outputs


class MHAM(nn.Module):

	def __init__(self,fpn_sizes, M,num_heads,att_drop):
		super(MHAM, self).__init__()
		self.fpn = PyramidFeatures(dim=fpn_sizes[-1],num_heads=num_heads,att_drop=att_drop)
		self.ca = Attentions(channel_size=M)
		self.prjo = nn.Linear(fpn_sizes[-1], fpn_sizes[-1])

	def forward(self, x):
		out = self.fpn(x)
		x = self.ca(out)
		x = self.prjo(x)
		return x


class BPP(nn.Module):
	def __init__(self, epsilon):
		super(BPP, self).__init__()
		self.epsilon = epsilon

	def forward(self, features1, features2):
		# unify the size of width and height
		B, C, H, W = features1.size()
		_, M, AH, AW = features2.size()

		# match size
		if AH != H or AW != W:
			features2 = F.upsample_bilinear(features2, size=(H, W))

		# essential_matrix: (B, M, C) -> (B, M * C)
		essential_matrix = (torch.einsum('imjk,injk->imn', (features2, features1)) / float(H * W)).view(B, -1)
		# nornalize
		essential_matrix = torch.sign(essential_matrix) * torch.sqrt(torch.abs(essential_matrix) + self.epsilon)
		essential_matrix = F.normalize(essential_matrix, dim=-1)

		return essential_matrix


class attbpp(nn.Module):
	def __init__(self):
		super(attbpp, self).__init__()
		self.mutipatt=PartSamplingAttention(dim=640, query_dim=640,query_size=(7,7))

	def forward(self, features1, features2):
		agg_feat=self.mutipatt(features1,features2)

		return agg_feat

class MutipCrossAttention(nn.Module):
	def __init__(self, dim, query_dim, heads_dim=4,att_drop=0.2):
		super().__init__()
		self.q = nn.Linear(query_dim, query_dim)
		self.kv = nn.Linear(dim, dim * 2)
		self.num_heads = dim // heads_dim
		self.query_dim = query_dim
		self.dim = dim
		self.scale = heads_dim ** -0.5
		self.atten_pos = nn.Parameter(torch.zeros((1, self.num_heads, self.query_dim // self.num_heads,
												   self.dim // self.num_heads)))
		self.softmax = nn.Softmax(dim=-1)
		self.o = nn.Linear(query_dim, query_dim)
		self.dropout = nn.Dropout(att_drop)

	# def flops(self):
	# 	flops = 0
	# 	# Q projection
	# 	flops += self.num_tokens * self.dim * self.dim
	# 	# KV projection
	# 	flops += self.query_dim * self.dim
	# 	# Attention Map
	# 	flops += self.dim * self.num_tokens
	# 	# V
	# 	flops +=  self.num_tokens * self.dim
	# 	# Add Learnable Bias
	# 	flops += self.num_heads * self.num_parts * self.num_tokens
	# 	# Feature Weights
	# 	# Enhance Feature Map
	# 	flops += self.num_tokens * self.dim
	# 	# O
	# 	flops += self.dim * self.dim * self.num_tokens
	# 	return flops


	def forward(self, parts1, parts2):
		B, P, C = parts1.shape
		self.num_tokens = P

		parts1=self.q(parts1)
		parts2=self.kv(parts2)

		q = rearrange(parts1, 'b p c -> b c p')
		kv = rearrange(parts2, 'b p c -> b c p')

		q = rearrange(q, 'b (h hc) n -> b h hc n', h=self.num_heads)
		kv = rearrange(kv, 'b (kv h hc) p -> kv b h hc p', kv=2, h=self.num_heads)
		k, v = kv[0], kv[1]

		attention_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
		attention_weights = attention_weights + self.atten_pos

		attention_score = self.softmax(attention_weights)
		attention_score = self.dropout(attention_score)

		x = rearrange(attention_score @ v, 'b h hc n -> b (h hc) n')
		x=rearrange(x, 'b c p -> b p c')
		x = self.o(x)

		return x



class PartSamplingAttention(nn.Module):
	def __init__(self, dim, query_dim, query_size, num_parts=16, heads_dim=16, parts_base=0.,
				 att_drop=0.2, parts_drop=0.2):
		super().__init__()
		self.q = nn.Linear(query_dim, dim)
		self.kv = nn.Linear(dim, dim * 2)
		self.num_parts = num_parts
		self.num_heads = heads_dim
		self.query_dim = query_dim
		self.parts_base = parts_base
		self.parts_drop = int(self.num_parts * parts_drop)
		self.dim = dim
		if self.parts_base:
			self.parts_base_num = int(self.num_parts * self.parts_base)
			self.learnable_parts = nn.Parameter(torch.randn((1, self.parts_base_num, dim)))
			self.num_parts = self.num_parts + self.parts_base_num
			torch.nn.init.kaiming_normal_(self.learnable_parts)
		self.parts_attention = PartSE(self.num_parts - self.parts_drop)
		self.scale = heads_dim ** -0.5
		self.atten_pos = nn.Parameter(torch.zeros((1, self.num_heads, query_size[0] * query_size[1],
												   self.num_parts - self.parts_drop)))
		self.softmax = nn.Softmax(dim=-1)
		self.softmax2 = nn.Softmax(dim=-2)
		self.weights_scale = nn.Parameter(torch.tensor(0.1))
		self.o = nn.Linear(dim, dim)
		self.dropout = nn.Dropout(att_drop)
		self.attn_drop = att_drop

	def flops(self):
		flops = 0
		# Q projection
		flops += self.num_tokens * self.dim * self.dim
		# KV projection
		flops += self.parts_drop * self.query_dim * self.dim
		# Parts Attention
		flops += self.parts_attention.flops()
		# Attention Map
		flops += self.parts_drop * self.dim * self.num_tokens
		# V
		flops += self.parts_drop * self.num_tokens * self.dim
		# Add Learnable Bias
		flops += self.num_heads * self.num_parts * self.num_tokens
		# Feature Weights
		flops += self.parts_drop * self.num_tokens
		# Enhance Feature Map
		flops += self.num_tokens * self.dim
		# O
		flops += self.dim * self.dim * self.num_tokens
		return flops


	def forward(self, x, parts):
		x = rearrange(x, 'b c h w -> b (h w) c')

		q = rearrange(self.q(x), 'b n (h hc) -> b h n hc', h=self.num_heads)
		kv = rearrange(self.kv(parts), 'b p (kv h hc) -> kv b h p hc', kv=2, h=self.num_heads)
		k, v = kv[0], kv[1]

		attention_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale

		# # Drop Key
		# if self.training:
		# 	m_r = torch.ones_like(attention_weights) * self.attn_drop
		# 	attention_weights = attention_weights + torch.bernoulli(m_r) * -1e12

		attention_score = self.softmax(attention_weights)


		# Drop Attention
		attention_score = self.dropout(attention_score)

		x = rearrange(attention_score @ v, 'b h n hc -> b n (h hc)')
		# x = rearrange(attention_score @ (v+self.weights_scale*parts_attention.reshape(B,1,-1,1)), 'b h n hc -> b n (h hc)')

		x = self.o(x)
		x = rearrange(x,'b (h w) c -> b c h w ',h=7,w=7)

		return x


class SpatialChanelRetrospect(nn.Module):
	def __init__(self, dim, input_size, cross_layer=False, backbone_type='hier', top_ratio=0.2, agg_channel_num=1,drop_path=0.2,sc=False,st=False):
		super().__init__()
		self.input_size = input_size
		self.cross_layer = cross_layer
		self.agg_channel_num = agg_channel_num
		self.top_ratio = top_ratio
		self.sc=sc
		self.st=st
		if self.cross_layer:
			self.norm_list = nn.ModuleList()
			if self.st:
				self.ST = nn.ModuleList()
			# self.down_list = nn.ModuleList()
			stage_scale_list = [8, 4, 2, 1] if backbone_type == 'hier' else [1, 1, 1, 1]
			for stage in stage_scale_list:
				self.norm_list.append(LayerNorm(dim // stage, eps=1e-6, data_format="channels_first"))
			# 语义跃迁检查模块
			if self.sc:
				self.SC=nn.ModuleList()
			for i in range(len(stage_scale_list) - 1):
				low_dim = dim // stage_scale_list[i]
				low_size = (input_size[0] * stage_scale_list[i], input_size[1] * stage_scale_list[i])
				high_dim = dim // stage_scale_list[i + 1]
				high_size = (input_size[0] * stage_scale_list[i + 1], input_size[1] * stage_scale_list[i + 1])
				if self.st:
					self.ST.append(
						SemanticTransition(high_dim, low_dim, low_size)
					)
				if self.sc:
					# self.SC.append(SemanticConstraints(low_dim,dim,low_size,drop_path))
					self.SC.append(ConvDecoder(low_dim, dim, low_size, drop_path))

			if self.sc:
				# self.SC.append(SemanticConstraints(low_dim,dim,low_size,drop_path))
				self.SC.append(ConvDecoder(dim, dim, (7,7), drop_path))
		# self.down_list.append(DownPatch(input_resolution=low_size, input_dim=(int(low_dim * top_ratio) + agg_channel_num)))
		# self.down_list.append(
		# 	DownMerging(input_resolution=low_size, dim=low_dim)
		# )
		self.activation = nn.GELU()

	# def flops(self):
	# 	flops = 0
	# 	if self.cross_layer:
	# 		for norm, part_generation, mpsa in zip(self.norm_list, self.parts_generation_list, self.mpsa_list):
	# 			flops += part_generation.flops()
	# 			flops += mpsa.flops()
	# 	return flops

	def forward(self, x):
		if self.cross_layer:
			# 语义跃迁检查
			processed_x = []
			for i in range(4):
				processed_x.append(self.norm_list[i](x[i]))
			transition_outputs = []
			if self.sc:
				for i in range(4):
					out = self.SC[i](processed_x[-1],processed_x[i])
					transition_outputs.append(out)
			if self.st:
				prev_suppressed = None
				for i in reversed(range(3)):
					low_feat = processed_x[i]
					if prev_suppressed is None:
						high_feat = processed_x[-1]
					else:
						high_feat = prev_suppressed
					out_1 = self.ST[i](high_feat, low_feat)
					prev_suppressed = out_1
					# selected_feats_final=self.down_list[i](selected_feats_final)
					# transition_outputs.append(selected_feats_final)

					transition_outputs.append(out_1)
				transition_outputs=transition_outputs[::-1]
				transition_outputs.append(processed_x[-1])


		# prev_suppressed = None
			# for i in reversed(range(3)):
			# 	low_feat = processed_x[i]
			# 	if prev_suppressed is None:
			# 		high_feat = processed_x[-1]
			# 	else:
			# 		high_feat = prev_suppressed
			# 	suppressed_low_feat = self.semantic_transition_list[i](high_feat, low_feat)
			# 	prev_suppressed = suppressed_low_feat
			#
			# 	transition_outputs.append(suppressed_low_feat)
		# transition_outputs = [self.activation(feat) for feat in transition_outputs]
		else:
			x = x[-1]
			x = self.norm(x)
			transition_outputs = self.activation(x)
		return transition_outputs


class SemanticTransition(nn.Module):
	def __init__(self,low_dim, low_size):
		super().__init__()
		self.low_dim = low_dim
		self.low_size = low_size
		self.norm_high=LayerNorm(low_dim, eps=1e-6, data_format="channels_first")
		self.norm_low = LayerNorm(low_dim, eps=1e-6, data_format="channels_first")

	# def flops(self):
	# 	conv_flops = self.high_size[0] * self.high_size[1] * self.high_dim * self.low_dim
	# 	sim_flops = self.low_size[0] * self.low_size[1] * self.low_dim
	# 	return conv_flops + sim_flops

	def forward(self, high_feat, low_feat):

		high_norm = self.norm_high(high_feat)
		low_norm = self.norm_low(low_feat)
		high_norm = rearrange(high_norm, 'b c h w -> b c (h w)')
		high_norm = F.normalize(high_norm, p=2, dim=1)
		low_norm = rearrange(low_norm, 'b c h w -> b c (h w)')
		low_norm = F.normalize(low_norm, p=2, dim=1)

		B, C, H, W = low_feat.shape
		N=H*W

		# 自己层相似，相似度高的表示聚集在一起
		xcosin = torch.bmm(low_norm.transpose(-1, -2), low_norm)#b,n,n数值表示两个通道之间的相似度值大表示相似
		topk_values, _ = torch.topk(xcosin, k=int(N*0.1), dim=-1)
		self_similarity_map = topk_values.mean(dim=-1)
		self_similarity_map = rearrange(self_similarity_map, 'b (h w) -> b h w',h=H,w=W)

		#交叉层
		cross_similarity_map = torch.sum(high_norm * low_norm, dim=1)
		cross_similarity_map=rearrange(cross_similarity_map, 'b (h w) -> b h w', h=H, w=W)
		patch_size = 2
		cross_similarity_patched = rearrange(
			cross_similarity_map,
			'b (h ph) (w pw) -> b h w ph pw',
			ph=patch_size, pw=patch_size
		)
		patch_base = cross_similarity_patched.sum(dim=(-1, -2), keepdim=True)
		patch_sum = cross_similarity_patched.sum(dim=(-1, -2), keepdim=True)
		relative_contribution = cross_similarity_patched / (patch_sum + 1e-8)
		contribution_patched = patch_base * relative_contribution
		contribution_map = rearrange(
			contribution_patched,
			'b h w ph pw -> b (h ph) (w pw)',
			h=self.low_size[0] // patch_size,
			w=self.low_size[1] // patch_size
		)

		self_sim_norm = torch.sigmoid(self_similarity_map)
		contrib_norm = 1.0 - torch.sigmoid(contribution_map)


		subject_mask = self_sim_norm * contrib_norm
		subject_mask_flat = subject_mask.reshape(B, -1)
		subject_mask_flat = F.normalize(subject_mask_flat, p=2, dim=1)

		subject_mask_expand = subject_mask_flat.unsqueeze(dim=1)
		channel_weight = (low_norm * subject_mask_expand).sum(dim=-1)

		channel_weight = torch.sigmoid(channel_weight)

		suppressed_low_feat = low_feat * subject_mask.unsqueeze(1) * channel_weight.unsqueeze(-1).unsqueeze(-1)
		suppressed_low_feat = suppressed_low_feat + low_feat

		return suppressed_low_feat

class SemanticConstraints(nn.Module):

	def __init__(self, prev_dim, last_dim, size,drop_path=0.2):
		super().__init__()
		self.size=size
		# self.channel_align = nn.Linear(last_dim, prev_dim)
		# self.gated_cnn = GatedCNNBlock(dim=prev_dim, drop_path=drop_path)
		# self.msc = MSC(dim=prev_dim)

	def forward(self, prev_feat, last_feat):
		# last_feat = F.interpolate(
		# 	last_feat,
		# 	size=self.size,
		# 	mode='nearest'
		# )
		# last_feat=rearrange(last_feat,'b c h w -> b h w c')
		# prev_feat = rearrange(prev_feat, 'b c h w -> b h w c')

		# aligned_last_feat = self.channel_align(last_feat)
		# gated_feat = self.gated_cnn(prev_feat)

		# aligned_last_feat = rearrange(aligned_last_feat, 'b h w c -> b c h w')
		# gated_feat = rearrange(gated_feat, 'b h w c -> b c h w')
		# fused_feat = self.msc(aligned_last_feat,gated_feat)

		return fused_feat

class MFF(nn.Module):
	def __init__(self, dim, drop_path=0.1,layer_scale_init_value=1e-6):
		super().__init__()
		self.dim = dim


		self.lska = LSKALight(dim, k_size=7)
		self.norm = LayerNorm(dim, eps=1e-6)
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
								   requires_grad=True) if layer_scale_init_value > 0 else None
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.pwconv = nn.Linear(dim, dim)

	def forward(self, x):



		layer = x[0]+x[1]+x[2]+x[3]
		layer2 = self.lska(layer)
		layer2 = layer + layer2

		layer2 = layer2.permute(0, 2, 3, 1)
		layer2 = self.norm(layer2)
		layer2 = self.pwconv(layer2)
		if self.gamma is not None:
			layer2 = self.gamma * layer2
		layer2 = layer2.permute(0, 3, 1, 2)

		layer2 = x[3] + self.drop_path(layer2)

		return layer2


class MultiPartRetrospect(nn.Module):
	def __init__(self, dim, input_size, parts_ratio=1, num_heads=4, att_drop=0.2, parts_drop=0.2,
				 pos=True, parts_base=0., cross_layer=False, backbone_type='hier'):
		super().__init__()
		self.input_size = input_size
		self.cross_layer = cross_layer
		self.num_parts = dim // parts_ratio
		if self.cross_layer:
			self.norm_list = nn.ModuleList()
			stage_scale_list = [8, 4, 2, 1] if backbone_type == 'hier' else [1,1,1,1]


			for i in range(2):
				self.norm_list.append(nn.LayerNorm(dim))
			self.parts_generation.append(
				PartSampling(dim, (input_size[0] * stage_scale_list[2], input_size[1] * stage_scale_list[2]),pos))
			self.mpsa.append(PartSamplingAttention(dim, dim, input_size, dim // (stage_scale_list[3] * parts_ratio),
															dim // num_heads, parts_base,
															att_drop, parts_drop))

		else:
			self.norm = nn.LayerNorm(dim)
			self.parts_generation = PartSampling(dim, input_size, self.num_parts, pos)
			self.mpsa = PartSamplingAttention(dim, dim, input_size, self.num_parts, dim // num_heads, parts_base)
		self.activation = nn.GELU()


	def flops(self):
		flops = 0
		if self.cross_layer:
			for norm, part_generation, mpsa in zip(self.norm_list, self.parts_generation_list, self.mpsa_list):
				flops += part_generation.flops()
				flops += mpsa.flops()
		return flops


	def forward(self, x):
		for i in range(2):
			x[i] = self.norm_list[i](x[i])
		parts = self.parts_generation_list(x[-1])#传出Fi采样图，经过PS模块，而后输出
		out = self.mpsa_list(x[-1], parts)#最后一层特征图和Fi处理，输出
		x = self.activation(out)
		return x


class MaskAttention(nn.Module):
	def __init__(self, channels, size):
		"""
		Args:
			channels: 输入特征的通道数 (C)
			size: 输入特征图的空间尺寸，格式为 (height, width)
		"""
		super(MaskAttention, self).__init__()
		self.channels = channels
		self.height, self.width = size
		self.num_patches = self.height * self.width  # 计算展平后的序列长度 N = H*W

		# QKV 映射层
		self.query = nn.Linear(channels, channels)
		self.key = nn.Linear(channels, channels)
		self.value = nn.Linear(channels, channels)

		# --- 核心修改：将 Mask 改为可学习参数 ---
		# 形状定义为 (1, N, N)，其中 N = H*W
		# 初始化：初始化为全 0 或很小的正态分布，让模型自己学习
		# 这样初始状态相当于“不 mask 任何位置”，训练中慢慢学会抑制
		self.mask = nn.Parameter(torch.zeros(1, self.num_patches, self.num_patches))

		# 层归一化
		self.norm = nn.LayerNorm(channels)

	def forward(self, x):
		batch_size, channels, height, width = x.size()

		# 校验输入尺寸
		if channels != self.channels or height != self.height or width != self.width:
			raise ValueError(
				f"Input shape mismatch. Expected (B, {self.channels}, {self.height}, {self.width}), got {x.shape}")

		# 1. 维度重塑: (B, C, H, W) -> (B, N, C), 其中 N=H*W
		x = x.view(batch_size, channels, self.num_patches).permute(0, 2, 1)

		# 2. 计算 Q, K, V
		Q = self.query(x)
		K = self.key(x)
		V = self.value(x)

		# 3. 计算缩放点积注意力
		scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, N, N)
		scores = scores / (self.channels ** 0.5)

		mask_probs = torch.sigmoid(self.mask)
		binary_mask = (mask_probs > 0.5).float()
		binary_mask = mask_probs + (binary_mask - mask_probs).detach()
		attention_mask = (binary_mask - 1.0) * 1e9

		# --- 核心修改：应用可学习的 Mask ---
		# self.mask 形状是 (1, N, N)
		# scores 形状是 (B, N, N)
		# 利用广播机制直接相加，无需考虑 batch_size
		scores = scores + attention_mask


		# 4. Softmax 与 加权求和
		attention_weights = F.softmax(scores, dim=-1)
		attention_output = torch.matmul(attention_weights, V)

		# 5. 残差连接与归一化
		attention_output = attention_output + x
		attention_output = self.norm(attention_output)

		# 6. 维度还原: (B, N, C) -> (B, C, H, W)
		attention_output = attention_output.permute(0, 2, 1).contiguous()
		return attention_output.view(batch_size, channels, height, width)



class Att(nn.Module):
	def __init__(self, dim,num_classes=23, nparts=4):
		super(Att, self).__init__()

		self.nparts = nparts
		self.dim = dim
		nlocal_channels_norm = self.dim // self.nparts
		reminder = self.dim % self.nparts
		nlocal_channels_last = nlocal_channels_norm
		if reminder != 0:
			nlocal_channels_last = nlocal_channels_norm + reminder
		fc_list = []
		separations = []
		sep_node = 0
		for i in range(self.nparts):
			if i != self.nparts - 1:
				sep_node += nlocal_channels_norm
				fc_list.append(nn.Linear(nlocal_channels_norm, num_classes))
			else:
				sep_node += nlocal_channels_last
				fc_list.append(nn.Linear(nlocal_channels_last, num_classes))
			separations.append(sep_node)
		self.fclocal = nn.Sequential(*fc_list)
		self.separations = separations
		self.fc = nn.Linear(self.dim, num_classes)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


	def forward(self, x):
		nsamples, nchannels, height, width = x.shape

		xview = x.view(nsamples, nchannels, -1)
		xnorm = xview.div(xview.norm(dim=-1, keepdim=True) + eps)
		xcosin = torch.bmm(xnorm, xnorm.transpose(-1, -2))

		attention_scores = []
		for i in range(self.nparts):
			if i == 0:
				xx = x[:, :self.separations[i]]
			else:
				xx = x[:, self.separations[i - 1]:self.separations[i]]
			xx_pool = self.avgpool(xx).flatten(1)
			attention_scores.append(self.fclocal[i](xx_pool))
		xlocal = torch.stack(attention_scores, dim=0)

		xmaps = x.clone().detach()

		# for global
		xpool = self.avgpool(x)
		xpool = torch.flatten(xpool, 1)
		xglobal = self.fc(xpool)

		return [xglobal, xlocal, xcosin, xmaps]


class CAM(nn.Module):
	def __init__(self, channels, reduction_ratio):
		super(CAM, self).__init__()
		self.channels = channels
		self.reduction_ratio = reduction_ratio  # 控制MLP瓶颈结构的压缩程度
		self.shard_mlp = nn.Sequential(
			nn.Linear(in_features=self.channels, out_features=self.channels // self.reduction_ratio, bias=True),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=self.channels // self.reduction_ratio, out_features=self.channels, bias=True)
		)
		self.maxpool2d = nn.AdaptiveMaxPool2d(output_size=1)  # 对H,W直接池化到只剩1个元素
		self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=1)

	def forward(self, x):
		max_pool = self.maxpool2d(x)  # 自适应2d最大池化，直接池化到只剩下一个元素
		avg_pool = self.avgpool2d(x)  # 自适应2d平均池化，直接池化到只剩下一个元素
		batch, channels, _, _ = x.size()  # 确定传过来的batch和channels, H, W 已经确定是1所以没必要获取

		# 先把维度调整到(b,c)以输入mlp，然后再把维度调回来
		max_pool_after_mlp = self.shard_mlp(max_pool.view(batch, channels)).view(batch, channels, 1, 1)
		avg_pool_after_mlp = self.shard_mlp(avg_pool.view(batch, channels)).view(batch, channels, 1, 1)

		channel_attention = torch.sigmoid(max_pool_after_mlp + avg_pool_after_mlp)
		output = channel_attention * x

		return output


class SAM(nn.Module):
	def __init__(self, bias=False):
		super(SAM, self).__init__()
		self.bias = bias
		self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=self.bias)

	def forward(self, x):
		# torch.max(x,1)：每个块先在channel维度上找到最大的值，返回（值，索引），然后[0]把值取出来获得一个(N,H,W)的图，接着用unsqueeze加回channel维度为1
		max = torch.max(x, 1)[0].unsqueeze(1)
		avg = torch.mean(x, 1).unsqueeze(1)  # mean直接返回值不返回索引
		concat = torch.cat((max, avg), dim=1)  # 沿着通道维度拼起来
		output = self.conv(concat)
		output = torch.sigmoid(output) * x
		return output


class CBAM(nn.Module):
	def __init__(self, channels, reduction_ratio, res_connect=False):
		super(CBAM, self).__init__()
		self.channels = channels
		self.reduction_ratio = reduction_ratio
		self.res_connect = res_connect
		self.sam = SAM(bias=False)
		self.cam = CAM(channels=self.channels, reduction_ratio=self.reduction_ratio)

	def forward(self, x0,x1):
		att0 = self.cam(x1)
		att1 = self.sam(x1)
		output = x0 * att0 * att1

		if self.res_connect:
			return output + x0
		else:
			return output


class SG(nn.Module):
	def __init__(self, dim):
		super(SG, self).__init__()
		self.guide_semantic_decoup = nn.Sequential(
			BasicConv(dim, dim // 4, kernel_size=1),
			LSKA(dim // 4, k_size=7),
			BasicConv(dim // 4, dim // 4, kernel_size=3, pad=1)
		)
		self.guide_weight_gen = nn.Sequential(
			nn.Conv2d(dim // 4, dim, kernel_size=1),
			nn.Sigmoid(),
		)
		self.guide_strength = nn.Parameter(torch.tensor(0.5), requires_grad=True)

	def forward(self, x0,x1):
		guide_semantic = self.guide_semantic_decoup(x1)
		guide_weight = self.guide_weight_gen(guide_semantic)
		guided_fusion_feat = x0 * (1 + self.guide_strength * guide_weight)

		return guided_fusion_feat



class CrossAttention(nn.Module):
	def __init__(self, d_model, n_heads=8):
		"""
        初始化 Cross Attention 模块
        参数:
            d_model: 输入的特征维度
            n_heads: 多头注意力的头数
        """
		super(CrossAttention, self).__init__()
		assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

		self.d_model = d_model
		self.n_heads = n_heads
		self.d_k = d_model // n_heads  # 每个头的维度

		# 定义 Q、K、V 的线性变换层
		self.W_q = nn.Linear(d_model, d_model)  # Query 的线性变换
		self.W_k = nn.Linear(d_model, d_model)  # Key 的线性变换
		self.W_v = nn.Linear(d_model, d_model)  # Value 的线性变换
		self.W_o = nn.Linear(d_model, d_model)  # 输出线性变换
		self.ca = CAM(d_model,16)
		self.sa = SAM()
		def forward(self, query, kv, mask=None):
		"""
        前向传播
        参数:
            query: 查询序列，形状 [batch_size, query_len, d_model]
            key: 键序列，形状 [batch_size, key_len, d_model]
            value: 值序列，形状 [batch_size, key_len, d_model]
            mask: 可选的注意力掩码，形状 [batch_size, query_len, key_len]
        返回:
            输出: 经过 Cross Attention 的结果，形状 [batch_size, query_len, d_model]
        """
		H=query.size(2)
		W=query.size(3)
		q = self.ca(query)
		k = self.sa(kv)
		v = self.sa(kv)

		key = rearrange(kv, 'b c h w -> b (h w) c')
		value = rearrange(kv, 'b c h w -> b (h w) c')
		query = rearrange(query, 'b c h w -> b (h w) c')
		k = rearrange(k, 'b c h w -> b (h w) c')
		v = rearrange(v, 'b c h w -> b (h w) c')
		q = rearrange(q, 'b c h w -> b (h w) c')

		key = 0*k + key
		value = 0*v + value
		query = 0*q + query

		batch_size = query.size(0)

		Q = self.W_q(query)
		K = self.W_k(key)
		V = self.W_v(value)

		Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
		K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
		V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

		scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)

		attn_weights = F.softmax(scores, dim=-1)
		attn_output = torch.matmul(attn_weights, V)

		attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
		output = self.W_o(attn_output)

		output = rearrange(output, 'b (h w) c -> b c h w',h=H,w=W)

		return output  # 返回输出和注意力权重（用于可视化或调试）
