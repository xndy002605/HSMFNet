# =====================================================================
# DATA LOADING & PREPROCESSING
# Purpose: Build dataloaders with:
#   SECTION 1: BUILD LOADER — dataset selection, transforms, augmentation, mixup
#   SECTION 2: NORMALIZATION — ImageNet-standard normalization parameters
# =====================================================================
import sys
from timm.data import Mixup, create_transform
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from settings.setup_functions import get_world_size
from utils.dataset import *


# =====================================================================
# SECTION 1: BUILD DATA LOADER
# Purpose: Construct train/test dataloaders with dataset-specific transforms
#          and optional mixup/cutmix augmentation.
# =====================================================================
def build_loader(config):
	train_transform, test_transform = build_transforms(config)

	train_set, test_set, num_classes = None, None, None
	if config.data.dataset == 'cub':
		root = os.path.join(config.data.data_root, 'CUB_200_2011')
		print(root)
		train_set = CUB(root, True, train_transform,download=True)
		test_set = CUB(root, False, test_transform)
		num_classes = 200

	elif config.data.dataset == 'cars':
		root = os.path.join(config.data.data_root, 'cars')
		train_set = Cars(root, True, train_transform)
		test_set = Cars(root, False, test_transform)
		num_classes = 196

	elif config.data.dataset == 'dogs':
		root = os.path.join(config.data.data_root, 'Dogs')
		train_set = Dogs(root, True, train_transform,download=True)
		test_set = Dogs(root, False, test_transform)
		num_classes = 120

	elif config.data.dataset == 'air':
		root = config.data.data_root
		train_set = Aircraft(root, True, train_transform)
		test_set = Aircraft(root, False, test_transform)
		num_classes = 100

	elif config.data.dataset == 'nabirds':
		root = os.path.join(config.data.data_root, 'nabirds')
		train_set = NABirds(root, True, train_transform)
		test_set = NABirds(root, False, test_transform)
		num_classes = 555

	elif config.data.dataset == 'pet':
		root = os.path.join(config.data.data_root, 'pets')
		train_set = OxfordIIITPet(root, True, train_transform)
		test_set = OxfordIIITPet(root, False, test_transform)
		num_classes = 37

	elif config.data.dataset == 'flowers':
		root = os.path.join(config.data.data_root, 'flowers')
		train_set = OxfordFlowers(root, True, train_transform)
		test_set = OxfordFlowers(root, False, test_transform)
		num_classes = 102

	elif config.data.dataset == 'food':
		root = config.data.data_root
		train_set = Food101(root, True, train_transform)
		test_set = Food101(root, False, test_transform)
		num_classes = 101
	elif config.data.dataset == 'FGSCR42':
		root = join(config.data.data_root,config.data.dataset)
		train_set = FGSC23Dataset(root, True, train_transform)
		test_set = FGSC23Dataset(root, False, test_transform)
		num_classes = config.model.num_classes
	elif config.data.dataset == 'FGSC23':
		root = join(config.data.data_root,config.data.dataset)
		train_set = FGSC23Dataset(root, True, train_transform)
		test_set = FGSC23Dataset(root, False, test_transform)
		num_classes = config.model.num_classes
	num_workers = 0
	if config.local_rank == -1:
		train_sampler = RandomSampler(train_set)
		test_sampler = SequentialSampler(test_set)
	else:
		train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(),
		                                   rank=config.local_rank, shuffle=True)
		test_sampler = DistributedSampler(test_set)
	train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=config.data.batch_size,
	                          num_workers=num_workers, drop_last=True, pin_memory=True)
	test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=config.data.batch_size,
	                         num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
	mixup_fn = None
	mixup_active = config.data.mixup > 0. or config.data.cutmix > 0.
	if mixup_active:
		mixup_fn = Mixup(
			mixup_alpha=config.data.mixup, cutmix_alpha=config.data.cutmix,
			label_smoothing=config.model.label_smooth, num_classes=num_classes)

	return train_loader, test_loader, num_classes, len(train_set), len(test_set), mixup_fn


def normalized():
	normalized_info = dict()
	normalized_info['standard'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
	return normalized_info
