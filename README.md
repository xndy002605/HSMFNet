# HSMFNet: Hierarchical Semantic-Guided Multi-Scale Fusion Network

Implementation of **"HSMFNet: A Hierarchical Semantic-Guided Multi-Scale Fusion Network for Fine-Grained Ship Classification in Remote Sensing Images"**.

## Repository Structure

```
HSMFNet/
├── main.py                  # Entry point: training loop, validation, checkpointing
│                               SECTION 1: Model Construction
│                               SECTION 2: Main Training Pipeline
│                                   2a. DATA LOADING
│                                   2b. MODEL BUILDING
│                                   2c. TRAINING LOOP
│                                   2d. VALIDATION
│                                   2e. SAVE OUTPUT
│                               SECTION 3: Training One Epoch
│                               SECTION 4: Validation (Testing)
├── setup.py                 # Configuration initialization
├── settings/
│   ├── defaults.py          # Default hyperparameters
│   └── setup_functions.py   # Environment setup utilities
├── models/
│   ├── build.py             # MODEL BUILDER
│   │                           SECTION 1: Backbone Selection
│   │                           SECTION 2: Pretrained Weight Loading
│   │                           SECTION 3: Freeze Backbone
│   ├── net.py               # HSMFNet CORE NETWORK
│   │                           STAGE 1: Backbone Feature Extraction
│   │                           STAGE 2: PMFE — Multi-Scale Fusion
│   │                           STAGE 3: SGCA — Semantic Calibration
│   │                           STAGE 4: CFL-HC — Hierarchical Classification
│   │                           STAGE 5: Return Output
│   ├── CHLHC.py             # CFL-HC: Coarse-Fine Linked Supervision Head
│   │                           FCSM: Fine-to-Coarse Semantic Mapping
│   │                           LOSS: Fine CE + Coarse CE + TreePathKL
│   ├── SGCA.py              # SGCA: Semantic-Guided Cross-Attention Calibration
│   ├── backbone/
│   │   ├── convnextv2.py    # ConvNeXtV2 backbone
│   │   ├── Swin_Transformer.py
│   │   ├── ResNet.py
│   │   └── Vision_Transformer.py
│   └── network/
│       ├── PMWF.py          # PMFE / PMWF: Progressive Multi-Scale Weighted Fusion
│       │                       CrossLayerFusion: core fusion + LKFE enhancement
│       └── LKFE.py          # LKFE: Large-Kernel Feature Enhancement
└── utils/
    ├── data_loader.py       # DATA LOADING & PREPROCESSING
    │                           SECTION 1: Build Data Loader
    │                           SECTION 2: Normalization
    ├── dataset.py           # Dataset classes (FGSC-23, FGSCR-42)
    ├── losses.py            # Loss functions
    ├── eval.py              # Evaluation: OA, Macro-F1, per-class precision/recall/F1
    ├── optimizer.py         # Optimizer configuration
    └── scheduler.py         # Learning rate scheduler
```

## Requirements

- Python
- PyTorch
- timm
- torchvision
- tqdm
- pandas
- tensorboard (optional)

## Quick Start

### 1. Prepare Dataset

Download FGSC-23 and FGSCR-42 datasets and organize as:
```
data/
├── FGSC23/
│   ├── train/
│   └── test/
└── FGSCR42/
    ├── train/
    └── test/
```

### 2. Training

```bash
python main.py
```

Key configuration parameters in `settings/defaults.py`:
- `data.dataset`: 'FGSC23' or 'FGSCR42'
- `data.batch_size`: batch size
- `train.epochs`: number of training epochs
- `model.type`: backbone type ('convnext', 'swin', 'resnet', 'vit')

### 3. Output

- Model checkpoints: saved to `config.data.log_path`
- Evaluation metrics: OA, Macro-F1, per-class precision/recall/F1
- TensorBoard logs: training curves, loss components
- Feature visualizations: t-SNE, activation maps (enable via flags)

## Citation

If you find this work useful, please cite our paper.

## License

This project is released for academic research purposes.
