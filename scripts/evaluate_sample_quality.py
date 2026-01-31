#!/usr/bin/env python3
"""Minimal quantitative evaluation for toy-crystals samples.

This script is intentionally simple: 2 metrics, nothing more.

Expected input format (for real data and generated samples):
  torch.save({"x_u8": uint8[N,1,64,64], "y_cat": int64[N], "y_cont": float32[N,4]}, ...)
which is the same format as scripts/build_dataset.py.

Metrics
  1) mass_w1 : 1D Wasserstein-1 distance between mean intensity distributions
  2) fft_radial_l2 : L2 distance between mean radial power spectra (normalised)

Output
  - prints a small table
  - optional CSV (semicolon-delimited for Dutch Excel settings)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import torch


def _load_x_u8(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    x = obj.get("x_u8", None)
    if x is None:
        raise KeyError(f"{path} does not contain key 'x_u8'")
    if x.dtype == torch.uint8:
        return x
    # allow float inputs for convenience
    x = x.to(torch.float32).clamp(0.0, 1.0)
    return (x * 255.0 + 0.5).to(torch.uint8)


def _subsample(x_u8: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    N = int(x_u8.shape[0])
    if n >= N:
        return x_u8
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(N, generator=g)[:n]
    return x_u8[idx]


def _w1_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Wasserstein-1 distance between 1D samples using sorted matching."""
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    n = min(int(a.numel()), int(b.numel()))
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    a = torch.sort(a).values
    b = torch.sort(b).values
    return float(torch.mean(torch.abs(a - b)).item())


def _radial_profile(power2d: torch.Tensor) -> torch.Tensor:
    """Radial average of a 2D power spectrum (assumes DC at centre)."""
    H, W = int(power2d.shape[-2]), int(power2d.shape[-1])
    cy, cx = H // 2, W // 2
    yy = torch.arange(H).view(H, 1).expand(H, W)
    xx = torch.arange(W).view(1, W).expand(H, W)
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.to(torch.int64)

    flat_r = r_int.flatten()
    flat_p = power2d.to(torch.float32).flatten()

    sum_by_r = torch.bincount(flat_r, weights=flat_p)
    count_by_r = torch.bincount(flat_r)
    prof = sum_by_r / torch.clamp(count_by_r, min=1)

    # sensible cutoff: Nyquist radius
    max_r = int(min(cy, cx))
    return prof[: max_r + 1]


def _fft_radial_profile(x_u8: torch.Tensor) -> torch.Tensor:
    """Return normalised mean radial power spectrum."""
    x = x_u8.to(torch.float32) / 255.0  # [N,1,H,W]
    x2 = x[:, 0]  # [N,H,W]

    f = torch.fft.fft2(x2)
    p = torch.abs(f) ** 2
    # shift DC to centre (avoid torch.fft.fftshift for compatibility)
    H, W = int(p.shape[-2]), int(p.shape[-1])
    p = torch.roll(p, shifts=(H // 2, W // 2), dims=(-2, -1))
    p_mean = p.mean(dim=0)

    prof = _radial_profile(p_mean)
    prof = prof / torch.clamp(prof.sum(), min=1e-12)
    return prof


def _metrics(real_u8: torch.Tensor, gen_u8: torch.Tensor) -> Dict[str, float]:
    # mass distribution: mean intensity per image
    real_mass = (real_u8.to(torch.float32) / 255.0).mean(dim=(1, 2, 3))
    gen_mass = (gen_u8.to(torch.float32) / 255.0).mean(dim=(1, 2, 3))
    mass_w1 = _w1_1d(real_mass, gen_mass)

    real_prof = _fft_radial_profile(real_u8)
    gen_prof = _fft_radial_profile(gen_u8)
    k = min(int(real_prof.numel()), int(gen_prof.numel()))
    fft_radial_l2 = float(torch.sqrt(torch.mean((real_prof[:k] - gen_prof[:k]) ** 2)).item())

    return {
        "mass_w1": mass_w1,
        "fft_radial_l2": fft_radial_l2,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--real", type=str, required=True, help="Held-out test set .pt file")
    p.add_argument(
        "--gen",
        action="append",
        default=[],
        help="Generated sample file as name=path.pt (can be passed multiple times)",
    )
    p.add_argument("--n-real", type=int, default=2048)
    p.add_argument("--n-gen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output (semicolon-delimited)")
    args = p.parse_args()

    if len(args.gen) == 0:
        raise SystemExit("Provide at least one --gen name=path.pt")

    real_u8 = _load_x_u8(args.real)
    real_u8 = _subsample(real_u8, args.n_real, seed=args.seed)

    results = []
    for spec in args.gen:
        if "=" not in spec:
            raise SystemExit("Each --gen must be in the form name=path.pt")
        name, path = spec.split("=", 1)
        gen_u8 = _load_x_u8(path)
        gen_u8 = _subsample(gen_u8, args.n_gen, seed=args.seed)
        m = _metrics(real_u8, gen_u8)
        results.append((name, m))

    # print
    print(f"real: {args.real} (n={int(real_u8.shape[0])})")
    for name, m in results:
        print(f"\n{name}")
        print(f"  mass_w1        : {m['mass_w1']:.6g}")
        print(f"  fft_radial_l2  : {m['fft_radial_l2']:.6g}")

    if args.csv is not None:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["name", "mass_w1", "fft_radial_l2", "real", "n_real", "n_gen", "seed"])
            for name, m in results:
                w.writerow([name, m["mass_w1"], m["fft_radial_l2"], args.real, args.n_real, args.n_gen, args.seed])
        print(f"\nWrote CSV -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
