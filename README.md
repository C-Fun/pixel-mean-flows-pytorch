# Pixel Mean Flows — PyTorch Fine-Tuning

unofficial PyTorch train/eval pipeline reimplementation of [One-step Latent-free Image Generation with Pixel Mean Flows (pixel Mean Flows / pMF)](https://arxiv.org/abs/2505.13447), faithfully ported from the [official JAX implementation](https://github.com/Lyy-iiis/pMF.git).

This baseline fine-tunes the pretrained `pMF-B/16` checkpoint on small image datasets (CIFAR-10, STL-10, Oxford Flowers-102) for reproduction and ablation experiments.

---

## What's implemented

| Component | Details |
|-----------|---------|
| MeanFlow loss | Compound velocity `V = u + (t−r)·du/dt` via `torch.autograd.forward_ad` |
| Guidance target | CFG guidance `v_g` via 2-forward pass (mirrors JAX `v_fn` + `v_cond_fn`) |
| Perceptual losses | LPIPS (VGG) + ConvNeXt-V2 feature matching, gated by `perceptual_max_t` |
| Optimizer | `torch.optim.Muon` (2D params) + AdamW (1D/3D+ params) |
| EMA | EDM-style schedule, three half-lives `[500, 1000, 2000]` kimg |
| Multi-GPU | DDP via `torchrun` |
| Datasets | CIFAR-10, STL-10, Oxford Flowers-102 |

---

## Setup

For the `pMF` folder, you can directly clone from the official repo: `git clone -b torch https://github.com/Lyy-iiis/pMF.git` 

```bash
pip install torch torchvision lpips transformers wandb tensorboard huggingface_hub
```

---

## Training

```bash
# Single GPU
python pMF_train.py --config configs/pMF_B_16_stl10.yml --workdir workdir/stl10

# Multi-GPU (e.g. 2 GPUs)
torchrun --nproc_per_node=2 pMF_train.py --config configs/pMF_B_16_stl10.yml --workdir workdir/stl10
```

Or use the provided launch script:

```bash
bash pMF_train.sh
```

### Available configs

| Config                                 | Dataset                 | Resolution |
|----------------------------------------|-------------------------|-----------|
| `configs/pMF_B_16_cifar10.yml`         | CIFAR-10                | 64×64 |
| `configs/pMF_B_16_stl10.yml`           | STL-10                  | 256×256 |
| `configs/pMF_B_16_flowers102_base.yml` | Oxford Flowers-102 w/o Perceptual losses | 256×256 |
| `configs/pMF_B_16_flowers102.yml`      | Oxford Flowers-102      | 256×256 |

---

## Key design notes

- **`loss=2.0` is expected**: `adp_wt` with `norm_p=1.0` normalizes each sample's contribution to ~1.0 when raw losses are large. Monitor `loss_u` / `loss_v` for training progress.
- **No gradient clipping**: Muon orthogonalizes gradients internally; AdamW adapts via second moment. Both handle large spatial-sum losses without explicit clipping.
- **Checkpoint format**: `{"model": state_dict, "ema_500": ..., "ema_1000": ..., "ema_2000": ...}`
- **Null class label**: index `num_classes` (e.g. 10 for CIFAR-10/STL-10, 102 for Flowers-102)

---

## Divergences from JAX

| Item | JAX | This repo |
|------|-----|-----------|
| Optimizer | `optax.contrib.muon` | `torch.optim.Muon` + AdamW |
| Vis sampling | raw params | EMA (2000 kimg) |
| Perceptual losses | LPIPS + ConvNeXt-V2 | same |

---

## Reference

```bib
@article{pixelmeanflows,
  title={One-step Latent-free Image Generation with Pixel Mean Flows},
  author={Lu, Yiyang and Lu, Susie and Sun, Qiao and Zhao, Hanhong and Jiang, Zhicheng and Wang, Xianbang and Li, Tianhong and Geng, Zhengyang and He, Kaiming},
  journal={arXiv preprint arXiv:2601.22158},
  year={2026}
}
```