"""
Fine-tune pMF-B/16 on CIFAR-10.
Implements the pixel MeanFlow training objective from scratch in PyTorch.
Launch with torchrun for multi-GPU:
  torchrun --nproc_per_node=<N> pMF_train.py --config configs/pMF_B_16_cifar10.yml --workdir <dir>
"""
import argparse
import copy
import math
import os
import shutil
import sys
import time
from types import SimpleNamespace

import lpips as lpips_lib
import torch
import torch.autograd.forward_ad as fwdAD
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import wandb
import yaml
from huggingface_hub import hf_hub_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import ConvNextV2Model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pMF"))
from models import pmfDiT  # noqa: E402


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path):
    """Load YAML config into a nested SimpleNamespace for dot-access."""
    with open(path) as f:
        d = yaml.safe_load(f)

    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        return obj

    return to_ns(d)


# ─── Distributed helpers ─────────────────────────────────────────────────────

def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1  # single-process fallback
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, dist.get_world_size()


def is_main(rank):
    return rank == 0


# ─── Sampling helpers ────────────────────────────────────────────────────────

def sample_tr(B, device, p_mean, p_std, data_proportion):
    """Sample t >= r from logit-normal. First data_proportion fraction: r=t (FM)."""
    t = torch.sigmoid(torch.randn(B, 1, 1, 1, device=device) * p_std + p_mean)
    r = torch.sigmoid(torch.randn(B, 1, 1, 1, device=device) * p_std + p_mean)
    fm_count = int(B * data_proportion)
    fm_mask = (torch.arange(B, device=device) < fm_count).view(B, 1, 1, 1)
    r = torch.where(fm_mask, t, r)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    return t, r, fm_mask


def sample_cfg_interval(B, fm_mask, device):
    """Per-sample CFG interval; FM samples use the full [0, 1] interval."""
    t_min = torch.rand(B, 1, 1, 1, device=device) * 0.5
    t_max = 0.5 + torch.rand(B, 1, 1, 1, device=device) * 0.5
    t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
    t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
    return t_min, t_max


def sample_cfg_scale(B, device, cfg_max):
    """Sample omega from power distribution (cfg_beta=1.0)."""
    u = torch.rand(B, 1, 1, 1, device=device)
    return torch.exp(u * math.log1p(cfg_max))


def edm_ema_beta(step, halflife_kimg, batch_size, world_size):
    """EDM-style EMA decay rate. Mirrors JAX ema_util.edm_schedule."""
    halflife_nimg = halflife_kimg * 1000
    halflife_nimg = min(halflife_nimg, step * batch_size * world_size * 0.05)
    return 0.5 ** (batch_size * world_size / max(halflife_nimg, 1e-8))


# ─── Perceptual aux loss ──────────────────────────────────────────────────────

def paired_random_resized_crop(imgs1, imgs2, size=224):
    """Apply same random resized crop to two image batches (per-sample). Mirrors JAX paired_random_resized_crop."""
    B = imgs1.shape[0]
    out1 = torch.empty(B, imgs1.shape[1], size, size, dtype=imgs1.dtype, device=imgs1.device)
    out2 = torch.empty_like(out1)
    for i in range(B):
        top, left, h, w = transforms.RandomResizedCrop.get_params(
            imgs1[i], scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)
        )
        out1[i] = TF.resized_crop(imgs1[i], top, left, h, w, [size, size],
                                   interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        out2[i] = TF.resized_crop(imgs2[i], top, left, h, w, [size, size],
                                   interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return out1, out2


def init_aux_models(mcfg, device):
    """Load frozen LPIPS and ConvNeXt-V2 feature extractors. Returns (lpips_model, convnext_model)."""
    lpips_model = None
    convnext_model = None
    if mcfg.lpips:
        lpips_model = lpips_lib.LPIPS(net="vgg").to(device).eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    if mcfg.convnext:
        convnext_model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
        convnext_model = convnext_model.to(device).eval()
        for p in convnext_model.parameters():
            p.requires_grad_(False)
    return lpips_model, convnext_model


# ─── Loss ────────────────────────────────────────────────────────────────────

def adp_wt(loss_per_sample, norm_p, norm_eps):
    w = (loss_per_sample + norm_eps) ** norm_p
    return loss_per_sample / w.detach()


@torch.no_grad()
def compute_v_g(net, z_t, t, v_t, omega, t_min, t_max, y, num_classes, fm_mask):
    """
    Compute guided target v_g and conditioned velocity v_c (for JVP tangent).
    Mirrors JAX guidance_fn + v_fn:
      Forward 1 (2B batch): [cond_full; uncond] → v_c_full, v_u  (JAX v_fn)
      Forward 2 (B):        cond with interval-masked omega → v_c  (JAX guidance_fn)
    """
    raw = net.module if isinstance(net, DDP) else net
    B = z_t.shape[0]
    t_flat = t.reshape(B)
    device = z_t.device

    # Forward 1: batch [cond_full; uncond] together — mirrors JAX v_fn
    y_null = torch.full((B,), num_classes, dtype=y.dtype, device=device)
    _, v2 = raw(
        torch.cat([z_t, z_t]),
        torch.cat([t_flat, t_flat]),
        torch.zeros(2 * B, device=device),
        torch.cat([omega.reshape(B), torch.ones(B, device=device)]),
        torch.zeros(2 * B, device=device),
        torch.ones(2 * B, device=device),
        torch.cat([y, y_null]),
    )
    v_c_full, v_u = v2[:B], v2[B:]
    v_g_fm = v_t + (1.0 - 1.0 / omega) * (v_c_full - v_u)

    # Forward 2: cond with interval-masked omega; h=0, t_min=0, t_max=1 (mirrors JAX v_cond_fn)
    w_interval = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
    _, v_c = raw(z_t, t_flat, torch.zeros(B, device=device), w_interval.reshape(B),
                 torch.zeros(B, device=device), torch.ones(B, device=device), y)
    v_g = v_t + (1.0 - 1.0 / w_interval) * (v_c - v_u)

    v_g = torch.where(fm_mask, v_g_fm, v_g)
    return v_g, v_c


def pmf_loss(net, x, y, num_classes, mcfg, aux_models=None):
    """mcfg: cfg.model SimpleNamespace. aux_models: (lpips_model, convnext_model) or None."""
    B = x.shape[0]
    device = x.device

    t, r, fm_mask = sample_tr(B, device, mcfg.p_mean, mcfg.p_std, mcfg.data_proportion)
    omega = sample_cfg_scale(B, device, mcfg.cfg_max)
    t_min, t_max = sample_cfg_interval(B, fm_mask, device)

    e = torch.randn_like(x) * mcfg.noise_scale
    z_t = (1.0 - t) * x + t * e
    v_t = (z_t - x) / t.clamp(min=0.05)

    v_g, v_c = compute_v_g(net, z_t, t, v_t, omega, t_min, t_max, y, num_classes, fm_mask)

    # CFG dropout: replace label → null, set v_g → v_t for dropped samples
    drop = torch.rand(B, device=device) < mcfg.class_dropout
    v_g = torch.where(drop.view(B, 1, 1, 1), v_t.detach(), v_g)
    y_train = torch.where(drop, torch.full_like(y, num_classes), y)
    v_g = v_g.detach()

    # JVP: tangent dz/dt=v_c, dt/dt=1, dr/dt=0 → dh/dt=1
    t_flat = t.reshape(B)
    h_flat = (t - r).reshape(B)

    with fwdAD.dual_level():
        z_dual = fwdAD.make_dual(z_t, v_c.detach())
        t_dual = fwdAD.make_dual(t_flat, torch.ones(B, device=device))
        h_dual = fwdAD.make_dual(h_flat, torch.ones(B, device=device))

        u_dual, v_dual = net(
            z_dual, t_dual, h_dual,
            omega.reshape(B), t_min.reshape(B), t_max.reshape(B),
            y_train,
        )
        u = fwdAD.unpack_dual(u_dual).primal
        du_dt = fwdAD.unpack_dual(u_dual).tangent
        v_out = fwdAD.unpack_dual(v_dual).primal  # aux output, primal only (JAX has_aux=True)

    V = u + (t - r) * du_dt.detach()

    loss_u_per = (V - v_g).pow(2).sum(dim=(1, 2, 3))
    loss_v_per = (v_out - v_g).pow(2).sum(dim=(1, 2, 3))

    per_sample = adp_wt(loss_u_per, mcfg.norm_p, mcfg.norm_eps) + adp_wt(loss_v_per, mcfg.norm_p, mcfg.norm_eps)

    lpips_per = torch.zeros(B, device=device)
    convnext_per = torch.zeros(B, device=device)
    lpips_model, convnext_model = aux_models if aux_models is not None else (None, None)
    if lpips_model is not None or convnext_model is not None:
        pred_x = z_t - t * u  # predicted clean image, gradients flow through u
        pred_crop, x_crop = paired_random_resized_crop(pred_x, x)
        t_mask = (t.reshape(B) < mcfg.perceptual_max_t).float()
        if lpips_model is not None:
            lpips_per = lpips_model(pred_crop, x_crop).reshape(B) * t_mask
            per_sample = per_sample + adp_wt(lpips_per, mcfg.norm_p, mcfg.norm_eps) * mcfg.lpips_lambda
        if convnext_model is not None:
            feat_pred = convnext_model(pred_crop).pooler_output   # (B, 1024)
            with torch.no_grad():
                feat_x = convnext_model(x_crop).pooler_output
            convnext_per = ((feat_pred - feat_x) ** 2).sum(dim=1) * t_mask
            per_sample = per_sample + adp_wt(convnext_per, mcfg.norm_p, mcfg.norm_eps) * mcfg.convnext_lambda

    loss = per_sample.mean()
    metrics = {
        "loss_u": loss_u_per.mean().item(),
        "loss_v": loss_v_per.mean().item(),
        "aux_lpips": lpips_per.mean().item(),
        "aux_convnext": convnext_per.mean().item(),
    }
    return loss, metrics


# ─── Sampling ────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_grid(ema_net, device, num_classes, img_size, scfg, step, writer):
    """ema_net: unwrapped EMA model (not DDP). scfg: cfg.sampling SimpleNamespace."""
    ema_net.eval()
    labels = (torch.arange(64) % num_classes).to(device)
    z = torch.randn(64, 3, img_size, img_size, device=device)

    t_steps = torch.linspace(1.0, 0.0, scfg.num_steps + 1, device=device)
    omega = torch.full((64,), scfg.omega, device=device)
    t_min = torch.full((64,), scfg.t_min, device=device)
    t_max = torch.full((64,), scfg.t_max, device=device)

    for i in range(scfg.num_steps):
        t_b = t_steps[i].expand(64)
        r_b = t_steps[i + 1].expand(64)
        h_b = t_b - r_b
        u, _ = ema_net(z, t_b, h_b, omega, t_min, t_max, labels)
        z = z - h_b[:, None, None, None] * u

    grid = torchvision.utils.make_grid(z, nrow=8, normalize=True, value_range=(-1, 1))
    wandb.log({"samples": wandb.Image(grid)}, step=step)
    writer.add_image("samples", grid, step)
    ema_net.train()


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    rank, world_size = setup_dist()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))

    if is_main(rank):
        os.makedirs(args.workdir, exist_ok=True)
        shutil.copy(args.config, os.path.join(args.workdir, "config.yml"))
        wandb.init(project=cfg.logging.wandb_project, name=cfg.logging.wandb_run_name,
                   config=yaml.safe_load(open(args.config)), dir=args.workdir)
        writer = SummaryWriter(log_dir=os.path.join(args.workdir, "tb"))
    else:
        writer = None

    num_classes = cfg.dataset.num_classes  # null label index = num_classes

    net_fn = getattr(pmfDiT, cfg.model.model_str)
    net = net_fn(input_size=cfg.model.img_size, in_channels=3,
                 num_classes=num_classes, eval_mode=False)

    ckpt_path = hf_hub_download(repo_id=cfg.checkpoint.hf_repo_id,
                                filename=cfg.checkpoint.hf_filename)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    if is_main(rank):
        print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    net = net.to(device).train()
    if world_size > 1:
        net = DDP(net, device_ids=[device.index])

    raw_net = net.module if isinstance(net, DDP) else net
    assert cfg.training.ema_type == "edm", f"Only EDM EMA supported, got {cfg.training.ema_type}"
    ema_nets = {k: copy.deepcopy(raw_net) for k in cfg.training.ema_val}
    for ema in ema_nets.values():
        ema.eval()

    aux_models = init_aux_models(cfg.model, device)
    if is_main(rank):
        print(f"Aux models: lpips={cfg.model.lpips}, convnext={cfg.model.convnext}")

    transform = transforms.Compose([
        transforms.Resize(cfg.model.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if is_main(rank):
        if cfg.dataset.name == "cifar10":
            torchvision.datasets.CIFAR10(cfg.dataset.data_root, train=True, download=True)
        elif cfg.dataset.name == "stl10":
            torchvision.datasets.STL10(cfg.dataset.data_root, split="train", download=True)
        elif cfg.dataset.name == "flowers102":
            torchvision.datasets.Flowers102(cfg.dataset.data_root, split="train", download=True)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    if world_size > 1:
        dist.barrier()
    if cfg.dataset.name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            cfg.dataset.data_root, train=True, transform=transform, download=False,
        )
    elif cfg.dataset.name == "stl10":
        dataset = torchvision.datasets.STL10(
            cfg.dataset.data_root, split="train", transform=transform, download=False,
        )
    elif cfg.dataset.name == "flowers102":
        dataset = torchvision.datasets.Flowers102(
            cfg.dataset.data_root, split="train", transform=transform, download=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True,
    )

    # Muon for strictly 2D params (weight matrices); AdamW for everything else (1D, 3D+ embeddings)
    muon_params = [p for p in net.parameters() if p.ndim == 2]
    adam_params = [p for p in net.parameters() if p.ndim != 2]
    optimizer = torch.optim.Muon(muon_params, lr=cfg.training.lr, momentum=0.95, weight_decay=0.0)
    adam_optimizer = torch.optim.AdamW(
        adam_params, lr=cfg.training.lr,
        betas=(0.9, cfg.training.adam_b2), weight_decay=0.0,
    )

    step = 0
    epoch = 0
    data_iter = iter(loader)
    t_last = time.perf_counter()
    step_last = 0

    while step < cfg.training.train_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        adam_optimizer.zero_grad()
        loss, metrics = pmf_loss(net, x, y, num_classes, cfg.model, aux_models)
        loss.backward()
        optimizer.step()
        adam_optimizer.step()
        step += 1

        # EMA update (all ranks; all produce identical results after DDP grad sync)
        for kimg, ema in ema_nets.items():
            beta = edm_ema_beta(step, kimg, cfg.training.batch_size, world_size)
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), raw_net.parameters()):
                    ep.lerp_(p, 1.0 - beta)

        if is_main(rank) and step % cfg.training.log_every == 0:
            t_now = time.perf_counter()
            sps = (step - step_last) / (t_now - t_last)
            t_last, step_last = t_now, step
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d}  loss={loss.item():.4f}  "
                  f"loss_u={metrics['loss_u']:.4f}  loss_v={metrics['loss_v']:.4f}  "
                  f"aux_lpips={metrics['aux_lpips']:.4f}  aux_convnext={metrics['aux_convnext']:.4f}  "
                  f"{sps:.2f} step/s")
            wandb.log({"loss": loss.item(), **metrics, "lr": lr, "step_per_sec": sps}, step=step)
            writer.add_scalar("loss", loss.item(), step)
            for k, v in metrics.items():
                writer.add_scalar(k, v, step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("step_per_sec", sps, step)

        if is_main(rank) and step % cfg.training.sample_every == 0:
            best_ema = ema_nets[max(ema_nets.keys())]
            sample_grid(best_ema, device, num_classes, cfg.model.img_size,
                        cfg.sampling, step, writer)

        if is_main(rank) and step % cfg.training.ckpt_every == 0:
            path = os.path.join(args.workdir, f"ckpt_{step:07d}.pt")
            ckpt_state = {"model": raw_net.state_dict()}
            for kimg, ema in ema_nets.items():
                ckpt_state[f"ema_{kimg}"] = ema.state_dict()
            torch.save(ckpt_state, path)
            print(f"Saved checkpoint: {path}")

    if is_main(rank):
        writer.close()
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Directory for checkpoints, logs, and config backup")
    main(parser.parse_args())