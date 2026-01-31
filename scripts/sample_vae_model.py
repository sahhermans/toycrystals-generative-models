#!/usr/bin/env python3
"""Sample images from the trained VAE using different priors.

Supported priors:
  - standard  : z ~ N(0, I)
  - diffusion : z ~ diffusion prior in (normalised) latent space
  - mop       : mixture-of-posteriors baseline (sample z from q(z|x) for real x)

Outputs:
  - .pt file compatible with scripts/build_dataset.py (dict with x_u8, y_cat, y_cont)
  - optional PNG grid for quick inspection
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from toycrystals.disk_data import ToyCrystalsDiskDataset
from toycrystals.models.vae import CondVAE, VAE
from toycrystals.models.diffusion_prior import DiffusionPriorFiLM, DiffusionPrior, DiffusionSchedule


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _sample_conditions(
    *,
    n: int,
    n_types: int,
    y_cont_dim: int,
    mode: str,
    theta_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "grid":
        y_cat = torch.tensor([i % n_types for i in range(n)], device=device, dtype=torch.int64)
        thetas = torch.linspace(0.0, float(theta_max), steps=n, device=device)
    elif mode == "random":
        y_cat = torch.randint(0, n_types, size=(n,), device=device, dtype=torch.int64)
        thetas = torch.rand((n,), device=device) * float(theta_max)
    else:
        raise ValueError("mode must be 'grid' or 'random'")

    y_cont = torch.zeros((n, y_cont_dim), device=device, dtype=torch.float32)
    if y_cont_dim > 1:
        y_cont[:, 1] = thetas
    return y_cat, y_cont


@torch.no_grad()
def _save_png_grid(x: torch.Tensor, out_png: str, n_show: int = 36) -> None:
    import matplotlib.pyplot as plt

    n = min(int(x.shape[0]), int(n_show))
    side = int(round(math.sqrt(n)))
    side = max(1, side)

    fig, axes = plt.subplots(side, side, figsize=(side, side))
    if side == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for i in range(side * side):
        ax = axes_list[i]
        if i < n:
            ax.imshow(x[i, 0].detach().cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


@torch.no_grad()
def _sample_standard_prior(
    vae: CondVAE | VAE,
    *,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    uncond: bool,
) -> torch.Tensor:
    z = torch.randn((int(y_cat.shape[0]), vae.z_dim), device=y_cat.device)
    if uncond:
        return vae.decode(z)
    return vae.decode(z, y_cat, y_cont)


@torch.no_grad()
def _sample_diffusion_prior(
    vae: CondVAE,
    *,
    y_cat: torch.Tensor,
    y_cont: torch.Tensor,
    prior_ckpt: str,
    latent_cache: str,
    T: int,
    beta_start: float,
    beta_end: float,
    ddim_steps: int,
    prior_arch: str,
    t_emb_dim: int,
    width: int,
    n_blocks: int,
    y_cat_emb_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if prior_arch == "film":
        prior = DiffusionPriorFiLM(
            z_dim=vae.z_dim,
            n_types=vae.n_types,
            y_cont_dim=vae.y_cont_dim,
            t_emb_dim=t_emb_dim,
            width=width,
            n_blocks=n_blocks,
            y_cat_emb_dim=y_cat_emb_dim,
        ).to(device)
    elif prior_arch == "mlp":
        prior = DiffusionPrior(
            z_dim=vae.z_dim,
            n_types=vae.n_types,
            y_cont_dim=vae.y_cont_dim,
            t_emb_dim=t_emb_dim,
            width=width,
        ).to(device)
    else:
        raise ValueError("prior-arch must be 'film' or 'mlp'")

    prior.load_state_dict(torch.load(prior_ckpt, map_location=device))
    prior.eval()

    obj = torch.load(latent_cache, map_location="cpu")
    z_mean = obj["z_mean"].to(device)
    z_std = obj["z_std"].to(device)

    sched = DiffusionSchedule.linear(T=T, beta_start=beta_start, beta_end=beta_end, device=device)

    z_norm = sched.ddim_sample(prior, y_cat=y_cat, y_cont=y_cont, n_steps=int(ddim_steps), eta=0.0)
    z = z_norm * z_std + z_mean
    return vae.decode(z, y_cat, y_cont)


@torch.no_grad()
def _sample_mop(
    vae: CondVAE,
    *,
    data_path: str,
    n: int,
    mode: str,
    theta_max: float,
    pool_size: int,
    decode_with: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_gen, y_target_cat, y_target_cont)."""
    ds = ToyCrystalsDiskDataset(data_path)

    # Candidate pool
    pool_size = min(int(pool_size), len(ds))
    idx_pool = torch.randint(0, len(ds), size=(pool_size,), device=torch.device("cpu"))
    x_pool = ds.x_u8[idx_pool].to(torch.float32) / 255.0
    ycat_pool = ds.y_cat[idx_pool]
    ycont_pool = ds.y_cont[idx_pool]

    # Target conditions
    y_target_cat, y_target_cont = _sample_conditions(
        n=n,
        n_types=vae.n_types,
        y_cont_dim=vae.y_cont_dim,
        mode=mode,
        theta_max=theta_max,
        device=torch.device("cpu"),
    )

    if mode == "grid":
        # match: same type + nearest theta (index 1)
        idxs = []
        for i in range(n):
            t = int(y_target_cat[i].item())
            theta_t = float(y_target_cont[i, 1].item()) if vae.y_cont_dim > 1 else 0.0
            mask = (ycat_pool == t)
            if not bool(mask.any()):
                idxs.append(int(torch.randint(0, pool_size, (1,)).item()))
                continue
            pool_idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if vae.y_cont_dim > 1:
                dtheta = (ycont_pool[pool_idxs, 1] - theta_t).abs()
                best_local = int(torch.argmin(dtheta).item())
            else:
                best_local = int(torch.randint(0, pool_idxs.shape[0], (1,)).item())
            idxs.append(int(pool_idxs[best_local].item()))
        idx = torch.tensor(idxs, dtype=torch.long)
    else:
        # random: just pick random pool elements
        idx = torch.randint(0, pool_size, size=(n,), dtype=torch.long)

    x_sel = x_pool[idx].to(device)
    y_sel_cat = ycat_pool[idx].to(device)
    y_sel_cont = ycont_pool[idx].to(device)

    mu, logvar = vae.encode(x_sel, y_sel_cat, y_sel_cont)
    z = vae.reparameterise(mu, logvar)

    if decode_with == "target":
        x_gen = vae.decode(z, y_target_cat.to(device), y_target_cont.to(device))
        return x_gen, y_target_cat.to(device), y_target_cont.to(device)

    # "source"
    x_gen = vae.decode(z, y_sel_cat, y_sel_cont)
    return x_gen, y_sel_cat, y_sel_cont


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # VAE
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--n-types", type=int, default=4)
    p.add_argument("--y-cont-dim", type=int, default=4)
    p.add_argument("--uncond", action="store_true", help="If set, sample from an unconditional VAE.")

    # Sampling
    p.add_argument("--prior", type=str, default="standard", choices=["standard", "diffusion", "mop"])
    p.add_argument("--mode", type=str, default="random", choices=["random", "grid"], help="How to sample y.")
    p.add_argument("--n", type=int, default=2048, help="Number of samples to write to the .pt output")
    p.add_argument("--theta-max", type=float, default=math.pi / 3.0)

    # Output
    p.add_argument("--out-pt", type=str, default=None, help="Where to save samples as a .pt dict (x_u8,y_cat,y_cont)")
    p.add_argument("--out-png", type=str, default=None, help="Optional PNG grid for quick inspection")
    p.add_argument(
        "--png-n",
        type=int,
        default=36,
        help="How many samples to include in the PNG preview grid (default: 36).",
    )

    # Diffusion-prior options
    p.add_argument("--prior-ckpt", type=str, default="checkpoints/diffusion_prior_last.pt")
    p.add_argument("--latent-cache", type=str, default="data/latents_rotonly_mu.pt")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--prior-arch", type=str, default="film", choices=["film", "mlp"], help="Must match how you trained the prior")
    p.add_argument("--t-emb-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--y-cat-emb-dim", type=int, default=64)

    # MoP options
    p.add_argument("--data-path", type=str, default="data/toycrystals_train_rotonly.pt", help="Disk dataset for MoP")
    p.add_argument("--pool-size", type=int, default=4096)
    p.add_argument("--mop-decode", type=str, default=None, choices=["target", "source"], help="Decode z with target y or source y")

    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    _set_seed(args.seed, device)

    if args.uncond:
        vae: CondVAE | VAE = VAE(z_dim=args.z_dim).to(device)
    else:
        vae = CondVAE(z_dim=args.z_dim, n_types=args.n_types, y_cont_dim=args.y_cont_dim).to(device)

    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()

    # Default output paths
    os.makedirs("results", exist_ok=True)
    if args.out_pt is None:
        args.out_pt = os.path.join("results", f"vae_samples_{args.prior}_{args.mode}.pt")

    # Sample conditions (for standard / diffusion)
    if args.prior in {"standard", "diffusion"}:
        y_cat, y_cont = _sample_conditions(
            n=int(args.n),
            n_types=vae.n_types if not args.uncond else args.n_types,
            y_cont_dim=vae.y_cont_dim if not args.uncond else args.y_cont_dim,
            mode=args.mode,
            theta_max=float(args.theta_max),
            device=device,
        )

    if args.prior == "standard":
        x = _sample_standard_prior(vae, y_cat=y_cat, y_cont=y_cont, uncond=bool(args.uncond))
        y_out_cat = y_cat
        y_out_cont = y_cont

    elif args.prior == "diffusion":
        if args.uncond:
            raise ValueError("diffusion prior sampling requires the conditional VAE")
        x = _sample_diffusion_prior(
            vae,
            y_cat=y_cat,
            y_cont=y_cont,
            prior_ckpt=args.prior_ckpt,
            latent_cache=args.latent_cache,
            T=int(args.T),
            beta_start=float(args.beta_start),
            beta_end=float(args.beta_end),
            ddim_steps=int(args.ddim_steps),
            prior_arch=args.prior_arch,
            t_emb_dim=int(args.t_emb_dim),
            width=int(args.width),
            n_blocks=int(args.n_blocks),
            y_cat_emb_dim=int(args.y_cat_emb_dim),
            device=device,
        )
        y_out_cat = y_cat
        y_out_cont = y_cont

    elif args.prior == "mop":
        if args.uncond:
            raise ValueError("mop sampling is only implemented for the conditional VAE")
        decode_with = args.mop_decode
        if decode_with is None:
            decode_with = "target" if args.mode == "grid" else "source"

        x, y_out_cat, y_out_cont = _sample_mop(
            vae,
            data_path=args.data_path,
            n=int(args.n),
            mode=args.mode,
            theta_max=float(args.theta_max),
            pool_size=int(args.pool_size),
            decode_with=decode_with,
            device=device,
        )

    else:
        raise ValueError("Unknown prior")

    # Save in the same compact format as build_dataset.py
    x_u8 = (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).detach().cpu()
    out = {
        "x_u8": x_u8,
        "y_cat": y_out_cat.detach().cpu().to(torch.int64),
        "y_cont": y_out_cont.detach().cpu().to(torch.float32),
    }

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)
    print(f"Saved samples -> {out_pt}")

    if args.out_png is not None:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        _save_png_grid(x, str(out_png), n_show=int(args.png_n))
        print(f"Saved preview grid -> {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
