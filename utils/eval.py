import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import precision_recall_fscore_support


class Timer:
	def __init__(self):
		self.times = []
		self.start()
		self.avg = 0.
		self.count = 0.
		self.sum = 0.

	def start(self):
		self.tik = time.time()

	def stop(self):
		t = time.time() - self.tik
		self.times.append(t)
		self.sum += t
		self.count += 1
		self.avg = self.sum / self.count
		return self.times[-1]

	def cumsum(self):
		return np.array(self.times).cumsum().tolist()


def simple_accuracy(preds, labels):
	count = preds.shape[0]
	result = (preds == labels).sum()
	return result / count


def reduce_mean(tensor):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= get_world_size()
	return rt


def count_parameters(model):
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return params / 1000000


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
	save_state = {'model': model.state_dict(),
	              'optimizer': optimizer.state_dict(),
	              'lr_scheduler': lr_scheduler.state_dict(),
	              'max_accuracy': max_accuracy,
	              'scaler': loss_scaler.state_dict(),
	              'epoch': epoch,
	              'config': config}

	# save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
	save_path = os.path.join(config.data.log_path, "checkpoint.bin")
	torch.save(save_state, save_path)
	print("----- Saved model checkpoint to", config.data.log_path, '-----')


def save_preds(preds, y, all_preds=None, all_label=None, ):
	if all_preds is None:
		all_preds = preds.clone().detach()
		all_label = y.clone().detach()
	else:
		all_preds = torch.cat((all_preds, preds), 0)
		all_label = torch.cat((all_label, y), 0)
	return all_preds, all_label


def load_checkpoint(config, model,optimizer, scheduler, loss_scaler, log):
	if config.local_rank in [-1, 0]:
		print('-' * 18, f'Resuming form \'{config.model.resume} \''.center(42), '-' * 18)
	checkpoint = torch.load(config.model.resume, map_location='cpu')
	state_dicts = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
	state_dicts = {k.replace('_orig_mod.', ''): v for k, v in state_dicts.items()}
	msg = model.load_state_dict(state_dicts, strict=True)

	log.info(msg)
	max_accuracy = 0.0
	if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
		# optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		config.defrost()
		config.train.start_epoch = checkpoint['epoch'] + 1
		config.freeze()
		if 'scaler' in checkpoint:
			loss_scaler.load_state_dict(checkpoint['scaler'])
		if config.local_rank in [-1, 0]:
			print('-' * 10, f"Loaded Successfully '{config.model.resume}' Epoch {checkpoint['epoch'] + 1}".center(58),
			      '-' * 10)
		if 'max_accuracy' in checkpoint:
			max_accuracy = checkpoint['max_accuracy']

	del checkpoint
	torch.cuda.empty_cache()
	return max_accuracy


def eval_accuracy(all_preds, all_label, config):
	accuracy = simple_accuracy(all_preds, all_label)
	if config.local_rank != -1:# TODO: 分布式未定义
		dist.barrier(device_ids=[config.local_rank])
		val_accuracy = reduce_mean(accuracy)
	else:
		val_accuracy = accuracy
	return val_accuracy.item()


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
		if isinstance(loss, (list, tuple)):
			# 假设 loss 是 (loss_small, loss_big)
			loss_small, loss_big = loss

			# 缩放和反向传播
			self._scaler.scale(loss_small).backward(create_graph=create_graph)
			self._scaler.scale(loss_big).backward(create_graph=create_graph, retain_graph=True)
		else:
			# 如果 loss 不是列表或元组，直接处理
			self._scaler.scale(loss).backward(create_graph=create_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = ampscaler_get_grad_norm(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == math.inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
		                                                norm_type).to(device) for p in parameters]), norm_type)
	return total_norm


def compute_class_metrics(all_preds, all_labels, num_classes):
    
    
	if all_preds.is_cuda:
		all_preds = all_preds.cpu().numpy()  # GPU → CPU → NumPy
	else:
		all_preds = all_preds.numpy()  # CPU 张量直接 → NumPy
    
	if all_labels.is_cuda:
		all_labels = all_labels.cpu().numpy()  # 同理处理真实标签
	else:
		all_labels = all_labels.numpy()
	precision, recall, f1, _ = precision_recall_fscore_support(
		all_labels, all_preds, labels=np.arange(num_classes), average=None, zero_division=0
	)
	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
	}
def get_world_size():
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()
