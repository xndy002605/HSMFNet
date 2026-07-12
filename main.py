import os

import pandas as pd
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from timm.utils import AverageMeter, accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm import tqdm

from models.build import build_models, freeze_backbone
from setup import config, log
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler


try:
	from torch.utils.tensorboard import SummaryWriter
except:
	pass


# =====================================================================
# SECTION 1: MODEL CONSTRUCTION
# Purpose: Build the HSMFNet model, move to GPU, freeze backbone if needed,
#          and log model structure and parameter count.
# =====================================================================
def build_model(config, num_classes):
	model = build_models(config, num_classes)
	model.to(config.device)
	freeze_backbone(model, config.train.freeze_backbone)
	model_without_ddp = model
	n_parameters = count_parameters(model)

	config.defrost()
	config.model.num_classes = num_classes
	config.model.parameters = f'{n_parameters:.3f}M'
	config.freeze()
	if config.local_rank in [-1, 0]:
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(), rank=config.local_rank)
		log.save(model)
	return model, model_without_ddp


# =====================================================================
# SECTION 2: MAIN TRAINING PIPELINE
# Purpose: Orchestrates the full training workflow:
#   2a. DATA LOADING — build dataloaders, mixup, compute steps
#   2b. MODEL BUILDING — construct HSMFNet, optimizer, scheduler, criterion
#   2c. TRAINING LOOP — iterate epochs, call train_one_epoch
#   2d. VALIDATION — evaluate after each epoch, save best checkpoint
#   2e. SAVE OUTPUT — log metrics, TensorBoard, Markdown results table
# =====================================================================
def main(config):
	# Timer
	total_timer = Timer()
	prepare_timer = Timer()
	prepare_timer.start()
	train_timer = Timer()
	eval_timer = Timer()
	total_timer.start()
	# Initialize the Tensorboard Writer
	writer = None
	if config.write:
		try:
			writer = SummaryWriter(config.data.log_path)
		except:
			pass

	# --- 2a. DATA LOADING: Build train/test dataloaders with data augmentation ---
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn = build_loader(config)
	step_per_epoch = len(train_loader)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch

	# --- 2b. MODEL BUILDING: Construct HSMFNet, optimizer, scheduler, loss criterion ---
	model, model_without_ddp = build_model(config, num_classes)

	if config.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
		                                                  broadcast_buffers=False,
		                                                  find_unused_parameters=False)
	# backbone_low_lr = config.model.type.lower() == 'resnet'
	# optimizer = build_optimizer(config, model, backbone_low_lr)
	optimizer = build_optimizer(config, model,config.parameters.backbone_low_lr)
	loss_scaler = NativeScalerWithGradNormCount()
	scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# Determine criterion
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	if config.data.mixup > 0.:
		criterion = SoftTargetCrossEntropy()
	elif config.model.label_smooth:
		criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	if config.model.resume:
		best_acc = load_checkpoint(config, model,optimizer, scheduler, loss_scaler, log)
		best_epoch = config.train.start_epoch
		accuracy, loss = valid(config, model,test_loader, best_epoch, train_accuracy,writer,False,num_classes=num_classes)
		log.info(f'Epoch {best_epoch+1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
		         f'BA {best_acc:2.3f}    BE {best_epoch+1:3}    '
		         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:
			return
	if config.misc.throughput:
		throughput(test_loader, model, log, config.local_rank)
		return

	# Record result in Markdown Table
	mark_table = PMarkdownTable(log, ['Epoch', 'Current Acc', 'Best Acc',
									  'Best Epoch', 'Loss'], rank=config.local_rank)

	# End preparation
	torch.cuda.synchronize()
	prepare_time = prepare_timer.stop()
	PSetting(log, 'Training Information',
	         ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
	          'Warm Epochs'],
	         [train_samples, test_samples, total_batch_size,
	          f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
	         newline=2, rank=config.local_rank)

	# --- 2c. TRAINING LOOP: Iterate over epochs, train and validate ---
	sub_title(log, 'Start Training', rank=config.local_rank)
	for epoch in range(config.train.start_epoch, config.train.epochs):
		train_timer.start()
		if config.local_rank != -1:
			train_loader.sampler.set_epoch(epoch)
			torch.cuda.empty_cache()

		train_loss=0.
		if not config.misc.eval_mode:
			train_accuracy,train_loss = train_one_epoch(config, model, criterion, train_loader, optimizer,
			                                 epoch, scheduler, loss_scaler, mixup_fn, writer)
		train_timer.stop()

		# --- 2d. VALIDATION: Evaluate on test set, save best checkpoint ---
		eval_timer.start()
		if (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:
			accuracy, loss = valid(config, model,test_loader, epoch, train_accuracy, writer,False,num_classes=num_classes,train_loss=train_loss)
			torch.cuda.empty_cache()
			if config.local_rank in [-1, 0]:
				if best_acc < accuracy:
					best_acc = accuracy
					best_epoch = epoch + 1
					if config.write and epoch > 1 and config.train.checkpoint:
						save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
				log.info(f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: test_Accuracy {accuracy:2.3f}    '
				         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
				         f'test_Loss {loss:1.4f}    train_Loss: {train_loss:.4f}    train_Accuracy {train_accuracy * 100:2.2f}')
				if config.write:
					mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
					                     f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
			pass  # Eval
		eval_timer.stop()
		pass  # Train

	# --- 2e. SAVE OUTPUT: Log final metrics, close TensorBoard writer ---
	if writer is not None:
		writer.close()
	train_time = train_timer.sum / 60
	eval_time = eval_timer.sum / 60
	total_time = train_time + eval_time
	total_time_true = total_timer.stop()
	total_time_true = total_time_true/60
	PSetting(log, "Finish Training",
	         ['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Syncthing Time','Total Time'],
	         [f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time_true-total_time:.2f} min' ,f'{total_time_true:.2f} min'],
	         newline=2, rank=config.local_rank)


# =====================================================================
# SECTION 3: TRAINING ONE EPOCH
# Purpose: Single-epoch training loop with automatic mixed precision (AMP),
#          loss computation, gradient scaling, and accuracy tracking.
# =====================================================================
def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, scheduler, loss_scaler, mixup_fn=None,
                    writer=None):
	model.train()
	optimizer.zero_grad()

	step_per_epoch = len(train_loader)
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()
	epochs = config.train.epochs

	loss1_meter = AverageMeter()
	loss2_meter = AverageMeter()
	loss3_meter = AverageMeter()
	loss4_meter = AverageMeter()


	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Train {epoch + 1:^3}/{epochs:^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])
	all_preds, all_label= None, None

	for step, (x, y) in enumerate(train_loader):
		global_step = epoch * step_per_epoch + step
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		if mixup_fn:
			x, y_hot = mixup_fn(x, y)
		# else:
		# 	y_hot = F.one_hot(y, num_classes=config.model.num_classes).float()
		with torch.amp.autocast('cuda', enabled=config.misc.amp):
			if config.model.baseline_model:
				logits = model(x)
				#print(f"logits shape: {logits.shape}")
			else:
				logits = model(x,y)
		if mixup_fn and not config.parameters.enh_head:
			logits, loss, other_loss = loss_in_iters(logits, y, criterion)
		else:
			logits, loss, other_loss = loss_in_iters(logits, y, criterion)
		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
		                        parameters=model.parameters(), create_graph=is_second_order)


		optimizer.zero_grad()
		scheduler.step_update(global_step + 1)
		loss_scale_value = loss_scaler.state_dict()["scale"]

		# if mixup_fn is None:

		preds = torch.argmax(logits, dim=-1)
		all_preds, all_label = save_preds(preds, y, all_preds, all_label)


		torch.cuda.synchronize()

		if grad_norm is not None:
			norm_meter.update(grad_norm)
		scaler_meter.update(loss_scale_value)
		loss_meter.update(loss.item(), y.size(0))

		lr = optimizer.param_groups[0]['lr']
		if writer:
			# writer.add_scalar("train/loss", loss_meter.val, global_step)
			writer.add_scalar("train/lr", lr, global_step)
			writer.add_scalar("train/grad_norm", norm_meter.val, global_step)
			writer.add_scalar("train/scaler_meter", scaler_meter.val, global_step)
			if other_loss:
				try:
					loss1_meter.update(other_loss[0].item(), y.size(0))
					loss2_meter.update(other_loss[1].item(), y.size(0))
					loss3_meter.update(other_loss[2].item(), y.size(0))
					loss4_meter.update(other_loss[3].item(), y.size(0))

				except:
					pass
				writer.add_scalar("losses/t_loss", loss_meter.val, epoch+1)
				writer.add_scalar("losses/coarse_loss", loss1_meter.val, epoch+1)
				writer.add_scalar("losses/fine_loss", loss2_meter.val, epoch+1)
				writer.add_scalar("losses/centerCoarse_loss", loss3_meter.val, epoch + 1)
				writer.add_scalar("losses/vRelative_loss", loss4_meter.val, epoch + 1)

		# set_postfix require dic input
		p_bar.set_postfix(tra_loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		p_bar.update()

	# After Training an Epoch
	p_bar.close()
	train_accuracy = eval_accuracy(all_preds, all_label, config)
	return train_accuracy,loss_meter.avg


def loss_in_iters(output, targets, criterion):
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets), None
	else:
		logits, loss = output
		if not isinstance(loss, (list, tuple)):
			return logits, loss, None
		else:
			return logits, loss[0], loss[1:]

# =====================================================================
# SECTION 4: VALIDATION (TESTING)
# Purpose: Evaluate model on test set, compute OA, per-class precision/recall/F1,
#          save per-class metrics to CSV, log to TensorBoard, and optionally
#          save features for t-SNE visualization.
# =====================================================================
@torch.no_grad()
def valid(config, model,test_loader, epoch=-1, train_acc=0.0, writer=None, save_feature=False, num_classes=None,train_loss=0.0):
	if hasattr(model, 'return\_feats'):
		model.return_feats = save_feature

	criterion = torch.nn.CrossEntropyLoss()
	model.eval()
	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	saved_feature,saved_labels = [],[]
	computer_feature = []

	all_preds, all_label = None, None

	# class_acc_meters = {i: AverageMeter() for i in range(num_classes)}

	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		# y_hot = F.one_hot(y, num_classes=config.model.num_classes).float()

		with torch.amp.autocast('cuda', enabled=config.misc.amp):
			if config.model.baseline_model:
				output = model(x)
			else:
				output = model(x,y)

			if isinstance(output, dict):
				# 开启return\_feats时，模型返回字典
				logits = output["logits"]
				feat_before_head = output["feat_before_head"]
			else:
				# 关闭时和原逻辑完全一致，只返回logits
				logits = output
				feat_before_head = None

		# loss = criterion(logits, y.long())
		logits, loss, other_loss = loss_in_iters(logits, y, criterion)

		if save_feature and feat_before_head is not None:
			computer_feature.append(feat_before_head.detach().cpu())

		if save_feature:
			saved_feature.append(logits)
			saved_labels.append(y)
		acc= accuracy(logits, y, topk=(1, ))[0]  # 计算整体准确率
		if config.local_rank != -1:# TODO: 未考虑多卡情况
			acc = reduce_mean(acc)
		preds = torch.argmax(logits, dim=-1)
		# for class_idx in range(num_classes):
		# 	class_mask = (y == class_idx)
		# 	class_total = class_mask.sum().item()
		# 	if class_total == 0:
		# 		continue
		# 	class_correct = (preds[class_mask] == class_idx).sum().item()
		# 	class_acc = (class_correct / class_total) * 100.0
		# 	class_acc_meters[class_idx].update(class_acc, class_total)

		all_preds, all_label = save_preds(preds, y, all_preds, all_label)

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg),
						  test_loss="%2.5f" % loss_meter.avg,
						  train_acc="{:2.3f}".format(train_acc * 100))
		p_bar.update()
		pass
	if save_feature:

		intra, inter, ratio = calculate_feature_metrics(saved_feature, saved_labels)
		print(f"类内紧凑性: {intra:.4f}")
		print(f"类间距离:   {inter:.4f}")
		print(f"分离度比值: {ratio:.4f}")

		os.makedirs('visualize/saved_features',exist_ok=True)
		saved_feature = torch.cat(saved_feature, 0)
		saved_labels = torch.cat(saved_labels,0)
		torch.save(saved_feature,f'visualize/saved_features/{config.data.dataset}_f.pth')
		torch.save(saved_labels, f'visualize/saved_features/{config.data.dataset}_l.pth')
	p_bar.close()
	class_metrics = compute_class_metrics(all_preds, all_label, num_classes)

	if config.write:
		metrics_dir = os.path.join(config.data.log_path, "class_metrics")
		os.makedirs(metrics_dir, exist_ok=True) 
		all_metrics_path = os.path.join(metrics_dir, "class_metrics_all.csv") 

		small_data = {
			"epoch": [epoch + 1] * num_classes,  
			"class_idx": list(range(num_classes)), 
			# "accuracy": [round(class_acc_meters[i].avg, 2) for i in range(num_classes)],
			"precision": [round(p, 4) for p in class_metrics['precision'].tolist()],  
			"recall": [round(r, 4) for r in class_metrics['recall'].tolist()],
			"f1": [round(f, 4) for f in class_metrics['f1'].tolist()]
		}
		small_df = pd.DataFrame(small_data)
		if not os.path.exists(all_metrics_path):
			small_df.to_csv(all_metrics_path, index=False, encoding="utf-8")
		else:
			small_df.to_csv(all_metrics_path, mode='a', header=False, index=False, encoding="utf-8")


	if writer:
		writer.add_scalar("test/accuracy", acc_meter.avg, epoch + 1)
		# writer.add_scalar("loss/test", loss_meter.avg, epoch + 1)
		# writer.add_scalar("loss/train", train_loss, epoch + 1)
		writer.add_scalars("loss",
                   {
                       "train": train_loss,  # 训练loss曲线
                       "test": loss_meter.avg  # 测试loss曲线
                   }, 
                   epoch + 1)
		if config.local_rank in [-1, 0]:
			# for i in range(num_classes):
			# 	writer.add_scalar(f"test/accuracy_class_{i}", class_acc_meters[i].avg, epoch + 1)
			# for i, (p, r, f) in enumerate(
			# 		zip(class_metrics['precision'], class_metrics['recall'], class_metrics['f1'])):
			# 	writer.add_scalar(f"test/precision_class_{i}", p, epoch + 1)
			# 	writer.add_scalar(f"test/recall_class_{i}", r, epoch + 1)
			# 	writer.add_scalar(f"test/f1_class_{i}", f, epoch + 1)
			overall_precision = class_metrics['precision'].mean()
			overall_recall = class_metrics['recall'].mean()
			overall_f1 = class_metrics['f1'].mean()
    
			# 仅记录整体指标（不带类别索引）
			writer.add_scalar("test/precision", overall_precision, epoch + 1)
			writer.add_scalar("test/recall", overall_recall, epoch + 1)
			writer.add_scalar("test/f1", overall_f1, epoch + 1)



	return acc_meter.avg,loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, log, rank):
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return
import torch.nn.functional as F
def calculate_feature_metrics(feats, labels):
	"""
    计算类内紧凑性和类间距离
    Args:
        feats:   [N, D] Tensor, N个样本的D维特征向量 (建议先L2归一化)
        labels:  [N] Tensor, N个样本对应的真实类别ID
    Returns:
        intra_compactness: 类内紧凑性 (越小越好)
        inter_distance:    类间特征距离 (越大越好)
        ratio:             类间距离 / 类内紧凑性 (越大越好)
    """
	# 获取数据集中出现的所有类别
	feats = torch.cat(feats,dim=0)
	labels = torch.cat(labels,dim=0)
	feats = feats.float()

	feats = F.normalize(feats, p=2, dim=1)

	unique_labels = torch.unique(labels)
	num_classes = len(unique_labels)

	# 1. 计算每个类别的特征中心 (类均值向量)
	centers = []
	for c in unique_labels:
		mask = (labels == c)
		class_feats = feats[mask]
		center = class_feats.mean(dim=0)  # [D]
		centers.append(center)
	centers = torch.stack(centers)  # [C, D]

	# 2. 计算类内紧凑性: 每个样本到其所属类中心的平均距离
	intra_dists = []
	for i, c in enumerate(unique_labels):
		mask = (labels == c)
		class_feats = feats[mask]
		# 计算该类所有样本到该类中心的欧氏距离
		dist = torch.norm(class_feats - centers[i], p=2, dim=1)
		intra_dists.append(dist.mean())
	intra_compactness = torch.stack(intra_dists).mean().item()

	# 3. 计算类间距离: 所有类中心之间的平均距离
	# 使用 torch.cdist 高效计算 [C, C] 的距离矩阵
	inter_dists_matrix = torch.cdist(centers, centers, p=2)
	# 我们只需要上三角部分(剔除对角线的0和下三角的重复计算)
	triu_indices = torch.triu_indices(num_classes, num_classes, offset=1)
	inter_distance = inter_dists_matrix[triu_indices[0], triu_indices[1]].mean().item()

	# 4. 综合指标比值
	ratio = inter_distance / intra_compactness

	return intra_compactness, inter_distance, ratio
if __name__ == '__main__':
	main(config)
