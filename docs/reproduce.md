# Reproducing the reported results

This file collects the **exact commands** used for the runs referenced in `README.md`.

Notes:
- Commands are shown for **bash / Git Bash** with `\` line continuations.
- On Windows `cmd.exe` you would use `^` instead.

---

## 1) Build datasets

Train set (seed 0):

```bash
python scripts/build_dataset.py   --out data/toycrystals_train_rotonly_seed0.pt   --n-samples 50000   --img-size 64   --n-types 4   --seed 0
```

Held-out test set (seed 1):

```bash
python scripts/build_dataset.py   --out data/toycrystals_test_rotonly_seed1.pt   --n-samples 50000   --img-size 64   --n-types 4   --seed 1
```

---

## 2) Train VAE

```bash
python scripts/train_vae.py   --device cuda   --seed 0   --data-path data/toycrystals_train_rotonly_seed0.pt   --z-dim 32   --n-types 4   --y-cont-dim 4   --batch-size 128   --epochs 15   --lr 2e-3   --beta 3e-4
```

Outputs (by default):
- `checkpoints/vae_last.pt`

---

## 3) Train latent diffusion prior (reported)

```bash
python scripts/train_diffusion_prior.py   --device cuda   --seed 0   --data-path data/toycrystals_train_rotonly_seed0.pt   --vae-ckpt checkpoints/vae_last.pt   --z-dim 32   --n-types 4   --y-cont-dim 4   --z-target mu   --latent-cache data/latents_mu_train_seed0.pt   --rebuild-latents   --max-items 50000   --T 400   --beta-start 1e-4   --beta-end 0.07   --ddim-steps 400   --width 256   --batch-size 256   --lr 1e-4   --epochs 200
```

Outputs (by default):
- `checkpoints/diffusion_prior_last.pt`
- `data/latents_mu_train_seed0.pt` (cached latents)

---

## 4) Sample VAE with different priors

### 4.1 Standard prior

```bash
python scripts/sample_vae_model.py   --device cuda   --seed 0   --vae-ckpt checkpoints/vae_last.pt   --prior standard   --mode random   --n 2048   --out-pt results/gen_vae_standard.pt   --out-png results/gen_vae_standard_preview.png   --png-n 100
```

### 4.2 Diffusion prior (reported)

```bash
python scripts/sample_vae_model.py   --device cuda   --seed 0   --vae-ckpt checkpoints/vae_last.pt   --prior diffusion   --mode random   --prior-ckpt checkpoints/diffusion_prior_last.pt   --latent-cache data/latents_mu_train_seed0.pt   --T 400   --beta-start 1e-4   --beta-end 0.07   --ddim-steps 400   --width 256   --n 2048   --out-pt results/gen_vae_diff_ddim400.pt   --out-png results/gen_vae_diff_ddim400_preview.png   --png-n 100
```

### 4.3 MoP (mixture-of-posteriors) reference

```bash
python scripts/sample_vae_model.py   --device cuda   --seed 0   --vae-ckpt checkpoints/vae_last.pt   --prior mop   --mode random   --data-path data/toycrystals_train_rotonly_seed0.pt   --pool-size 50000   --mop-decode target   --n 2048   --out-pt results/gen_vae_mop.pt   --out-png results/gen_vae_mop_preview.png   --png-n 100
```

---

## 5) Direct VP-SDE (training + sampling; reported)

### Dataset filename note

The original VP-SDE training run used `data/toycrystals_train_rotonly.pt` (no `_seed0` suffix).
If you want to reproduce that *exact* path, either:

- build it directly:

```bash
python scripts/build_dataset.py   --out data/toycrystals_train_rotonly.pt   --n-samples 50000   --img-size 64   --n-types 4   --seed 0
```

- or copy the seed-0 training set:

```bash
cp data/toycrystals_train_rotonly_seed0.pt data/toycrystals_train_rotonly.pt
```


### 5.1 Training (actual run invocation)

```bash
python scripts/train_sde_score_model.py   --device cuda   --seed 0   --data-path data/toycrystals_train_rotonly.pt   --out-dir runs/sde_score/night_20260116_152825/exp14_ch96_lr1e-4_ep200_ema_pu05   --base-ch 96   --batch-size 128   --beta-max 30.0   --beta-min 0.1   --cond-ch 8   --ema-decay 0.999   --emb-dim 128   --epochs 200   --lr 1e-4   --n-types 4   --p-uncond 0.05   --sample-every 1000000   --sample-from-ema 1   --t-power 1.0   --time-ch 8   --y-cont-dim 4
```

### 5.2 Sampling (reported settings)

Reported settings: SDE sampler, steps 300, cfg 1.5, t_end 0.005, EMA on.

```bash
python scripts/sample_sde_score_model.py   --device cuda   --out-dir results   --ckpt runs/sde_score/night_20260116_152825/exp14_ch96_lr1e-4_ep200_ema_pu05/checkpoints/sde_score_model_last.pt   --sampler sde   --steps 300   --cfg 1.5   --t-end 0.005   --use-ema 1   --n 36   --out-path results/sde_samples_preview.png   --pt-out results/gen_sde.pt   --pt-n 2048   --pt-batch-size 64   --pt-seed 0
```

If you hit CUDA OOM during `.pt` generation, reduce `--pt-batch-size` (e.g. 32, 16, 8).

---

## 6) Evaluation

```bash
python scripts/evaluate_sample_quality.py   --real data/toycrystals_test_rotonly_seed1.pt   --gen vae_standard=results/gen_vae_standard.pt   --gen vae_diffusion=results/gen_vae_diff_ddim400.pt   --gen vae_mop=results/gen_vae_mop.pt   --gen sde=results/gen_sde.pt   --n-real 2048   --n-gen 2048   --csv results/quality_scores.csv
```
