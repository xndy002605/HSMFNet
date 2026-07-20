from torch import nn
from models.HybirdClsHead import HierarchicalClsHead
from models.backbone.utils import LayerNorm
from models.cores import SpatialChanelRetrospect, Att, CBAM, SemanticTransition, SG, CrossAttention
from models.network.mmf import CrossLayerFusion
from models.network.semic_f import Duble_fusion


# =====================================================================
# HSMFNet CORE NETWORK
# Architecture: ConvNeXtV2 Backbone → PMFE (Multi-Scale Fusion) → SGCA (Semantic
#               Calibration) → CFL-HC (Hierarchical Classification Head)
# Purpose: End-to-end fine-grained ship classification network.
# =====================================================================
class Net(nn.Module):
	def __init__(self, dim, input_size, focal_loss_alpha,fine2coarse,
				 backbone=None, head_drop=0.5,
				 num_classes=42, cross_layer=False,backbone_type='hier',
				 top_ratio=0.2,agg_channel_num=1,enh_head=True,sc=False,st=False,drop_path=0.1,return_feats=False
				 ):
		super(Net, self).__init__()
		self.input_size = (input_size//32,input_size//32) if backbone_type=='hier' else (input_size//16,input_size//16)
		self.dim = dim
		self.num_classes = num_classes
		self.activation = nn.GELU()
		self.return_feats = return_feats
		self.sc=sc
		self.st=st
		if st or sc:
			self.block = SpatialChanelRetrospect(self.dim, self.input_size,cross_layer,backbone_type,top_ratio,agg_channel_num,drop_path,sc,st)
		self.backbone_type = backbone_type
		self.focal_loss_alpha = focal_loss_alpha
		self.fine2coarse = fine2coarse
		self.enh_head = enh_head
		self.sc=sc
		self.st=st
		# self.mff=MFF(dim,drop_path)

		self.pooling = nn.AdaptiveAvgPool1d(1)
		if cross_layer:
			# new_dim = int(dim * 15 / 8) if self.backbone_type == 'hier' else int(dim * 4)
			# new_dim = (768+(int(384*top_ratio)+agg_channel_num)*4+(int(192*top_ratio)+agg_channel_num)*16+(int(96*top_ratio)+agg_channel_num)*64) if self.backbone_type == 'hier' else int(dim * 4)
			# new_dim=dim+384+192+96
			# self.norm = nn.LayerNorm(new_dim)
			new_dim = dim

			# self.norm_list = nn.ModuleList()
			# stage_scale_list = [8,4,2,1]
			# for stage in stage_scale_list:
			# 	# self.norm_list.append(LayerNorm(dim // stage, eps=1e-6, data_format="channels_first"))
			# 	self.norm_list.append(LayerNorm(dim // stage, eps=1e-6, data_format="channels_first"))


			if enh_head:
				# self.norm = nn.LayerNorm(dim, eps=1e-6)
				self.chlhc = CHLHC(dim=new_dim,fine2coarse=fine2coarse, num_classes=num_classes,head_drop=head_drop)
			else:
				# self.norm= nn.LayerNorm(dim, eps=1e-6)
				# self.head = nn.Linear(new_dim, num_classes)
				self.norm = nn.LayerNorm(new_dim, eps=1e-6)
				self.head = nn.Linear(new_dim, num_classes)
				self.head_drop = nn.Dropout(head_drop)
		else:
			self.norm = nn.LayerNorm(dim, eps=1e-6)
			self.head = nn.Linear(dim, num_classes)
			self.head_drop = nn.Dropout(head_drop)
		self.show = nn.Identity()
		self.apply(self.init_weights)
		self.backbone = backbone
		self.assess = False
		self.save_feature = None
		self.count = 0
		# self.lska=LSKA(dim,k_size=7)
		# self.sfe=SEFProcessor(
		# 	in_channels=dim,
		# 	num_classes=num_classes,
		# 	n_groups=4,
		# 	rho=0.1,
		# 	lambda_ent=0.01,
		# 	gamma=0.1
		# )

		self.pmwf=PMWF()
		self.sgca = SGCA(dim)


		# self.st=SemanticTransition(dim,(14,14))
		# self.sg = SG(dim)
		# self.cbam = CBAM(channels=dim, reduction_ratio=16, res_connect=True)
		# self.att = MaskAttention(channels=640, size=(14, 14))
		# self.safpa = SimpleFPA(dim,dim)
		# self.part_fusion = Part_fusion(dim, (7,7))
		# self.df = Duble_fusion(dim)
		# self.st = SemanticTransition(dim,(14,14))
		# self.down_conv = nn.Conv2d(640, 320, kernel_size=1)
		# self.final_norm = LayerNorm(320, eps=1e-6, data_format="channels_first")
		self.u3 = nn.Upsample(scale_factor=2, mode='bilinear')
		# self.msn = MSC(dim)
		# self.seg = SegMANDecoder(dim)
		# self.bpp = BPP(1e-12)
		# self.gnn = GatedCNNBlock(656)


	def init_weights(self, m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			nn.init.kaiming_normal_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		if hasattr(self, 'head'):
			nn.init.constant_(self.head.weight, 0)
			nn.init.constant_(self.head.bias, 0)
	def flops(self):
		flops = 0
		# Backbone
		flops += self.backbone.flops()
		# HPR
		flops += self.block.flops()
		# Delete Original Norm
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Delete Original Head
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Norm
		flops += self.dim * 15 / 8 * self.input_size[0] * self.input_size[0]
		# Multi-Grained Fusion
		flops += self.dim * self.input_size[0] * self.input_size[0]
		# Head
		flops += self.dim * 15 / 8 * self.num_classes
		return flops
	def forward(self, x,label=None):
		# ===== STAGE 1: BACKBONE FEATURE EXTRACTION =====
		# Extract multi-stage features from ConvNeXtV2 backbone
		x = self.backbone(x)

		# ===== STAGE 2: PMFE — PROGRESSIVE MULTI-SCALE FUSION =====
		# Fuse low-level details with high-level semantics via adaptive weighting
		final_feat = self.pmfe((x[0],x[1],x[2],x[3]))

		# ===== STAGE 3: SGCA — SEMANTIC-GUIDED CROSS-ATTENTION CALIBRATION =====
		# Use high-purity semantic features as anchor to calibrate fused features
		final_feat = self.sgca(x[3],final_feat)

		x[3] = x[3].mean([-2, -1])
		final_feat = final_feat.mean([-2,-1])
		feat_before_head = final_feat

		# ===== STAGE 4: CFL-HC — HIERARCHICAL CLASSIFICATION HEAD =====
		# Coarse-Fine Linked Supervision: coarse-grained semantic priors
		# constrain fine-grained classification to mitigate class imbalance
		if self.enh_head:
			final_feat=self.chlhc(final_feat,x[3],label)
		else:
			final_feat = self.norm(final_feat)
			final_feat = self.head_drop(final_feat)
			final_feat = self.head(final_feat)

		# final_feat=self.sfe(x[-1])

		# ===== STAGE 5: RETURN OUTPUT =====
		# Training mode: return logits for loss computation
		# Feature extraction mode: return logits + pre-head features for t-SNE
		if not self.return_feats:
			return final_feat
		else:
			return {
				"logits": final_feat,
				"feat_before_head": feat_before_head
			}