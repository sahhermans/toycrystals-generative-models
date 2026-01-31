# ToyCrystals — conditional generation on synthetic lattice images

This repo explores **conditional generation** on a synthetic *toy-crystals* dataset: periodic lattice images rendered as Gaussian “atoms”.

It implements and compares three approaches:

1. **VAE + standard prior** (baseline)
2. **VAE + learned diffusion prior in latent space** (best overall here)
3. **Direct score-based diffusion on images (VP-SDE)** (no VAE)

Diffusion/SDE notation is aligned with the MIT diffusion course lecture notes ([PDF](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)). See [`docs/reproduce.md`](docs/reproduce.md) for the exact commands used for the reported runs.

## Results

### Qualitative

<div align="center"><b>VAE reconstructions</b></div>
<p align="center">
  <img src="assets/vae_standard_prior/vae_recon.png" width="700" />
</p>

<div align="center"><b>Sampling comparison (same conditioning distribution)</b></div>

<div align="center">
<table>
  <tr>
    <td align="center">
      <b>VAE + standard prior</b><br/>
      <sub>z ~ N(0, I)</sub><br/>
      <img src="assets/vae_standard_prior/vae_standard_prior_sampling.png" width="450" />
    </td>
    <td align="center">
      <b>VAE + latent diffusion prior</b><br/>
      <sub>DDIM sampling in latent space</sub><br/>
      <img src="assets/vae_latent_diffusion_prior/vae_latent_diffusion_prior_sampling.png" width="450" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>VAE + MoP reference</b><br/>
      <sub>posterior-based baseline</sub><br/>
      <img src="assets/vae_standard_prior/vae_mop_sampling.png" width="450" />
    </td>
    <td align="center">
      <b>Direct VP-SDE</b><br/>
      <sub>reverse-SDE sampler</sub><br/>
      <img src="assets/score_based_diffusion/score_based_diffusion_samples.png" width="450" />
    </td>
  </tr>
</table>
</div>


### Quantitative (held-out test set)

Lower is better.

| method | mass_w1 ↓ | fft_radial_l2 ↓ |
|---|---:|---:|
| VAE + standard prior | 0.01148 | 0.00218 |
| VAE + diffusion prior (DDIM 400) | **0.00144** | **0.000313** |
| VAE + MoP (posterior-based reference) | 0.00307 | **0.000256** |
| VP-SDE (reverse-SDE sampler) | 0.00202 | 0.000584 |
| VP-SDE (probability-flow ODE) | 0.00811 | 0.00173 |

Takeaway: the **latent diffusion prior** provides the best overall match on these two simple metrics, while MoP remains a strong “anchored-to-data” reference.

## Dataset

Each sample is a single-channel image $x \in [0,1]^{1\times H\times W}$ containing a periodic lattice (Gaussian “atoms”).

We train conditionally on $c$, consisting of:

- a categorical **lattice type** $c_{\mathrm{cat}}$
- continuous **rotation / geometry features** $c_{\mathrm{cont}}$ (encoded as a small vector; e.g. sine/cosine-style features)

Datasets are stored as `.pt` tensors on disk (see `scripts/build_dataset.py`).

## Models

### Conditional VAE

We use a conditional VAE with encoder $q_\phi(z\mid x,c)$ and decoder $p_\theta(x\mid z,c)$, with a standard Normal prior $p(z)=\mathcal{N}(0,I)$.

**Objective (negative ELBO)**

$$
\mathcal{L}_{\mathrm{VAE}}(\theta,\phi) = \mathbb{E}_{q_\phi(z\mid x,c)}\big[-\log p_\theta(x\mid z,c)\big] + \beta\,\mathrm{KL}\!\big(q_\phi(z\mid x,c)\,\|\,\mathcal{N}(0,I)\big).
$$

Key point: even with a moderate $\beta$, the *aggregated posterior*

$$
q(z\mid c)=\mathbb{E}_{x\sim p_{\mathrm{data}}(x\mid c)}\left[q_\phi(z\mid x,c)\right]
$$

can be a poor match to $\mathcal{N}(0,I)$. This is why standard-prior sampling $z\sim\mathcal{N}(0,I)$ can produce “off-manifold” artefacts, even if reconstructions are good.

### Latent diffusion prior

Instead of sampling $z\sim\mathcal{N}(0,I)$, we train a diffusion model in **latent space** to approximate a better conditional prior $p_\psi(z\mid c)$.

We train on cached VAE latents (typically the posterior mean $z_0 = \mu_\phi(x,c)$).

**Forward noising process (DDPM form)**  
With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$:

$$
z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
$$

**Training objective (noise prediction)**

$$
\mathcal{L}_{\mathrm{diff}}(\psi)=
\mathbb{E}_{t,\varepsilon,z_0,c}\left[\lVert \varepsilon - \varepsilon_\psi(z_t,t,c) \rVert_2^2\right].
$$

Sampling uses DDIM-style updates (deterministic, fewer steps) or the full $T$-step process, producing $z_0\sim p_\psi(z\mid c)$ which is decoded by the VAE: $x\leftarrow p_\theta(x\mid z_0,c)$.

### Direct score-based diffusion (VP-SDE)

We also train a score model directly on images (no VAE) using the VP-SDE parameterisation:

$$
\mathrm{d}x = -\tfrac12\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}w_t.
$$

The reverse-time SDE has drift that depends on the score $\nabla_x\log p_t(x\mid c)$:

$$
\mathrm{d}x =
\left[-\tfrac12\beta(t)x - \beta(t)\nabla_x\log p_t(x\mid c)\right]\mathrm{d}t
+\sqrt{\beta(t)}\,\mathrm{d}\bar w_t.
$$

The associated probability-flow ODE is:

$$
\frac{\mathrm{d}x}{\mathrm{d}t} = -\tfrac12\beta(t)x - \tfrac12\beta(t)\nabla_x\log p_t(x\mid c).
$$

At sampling time we use either the reverse-SDE sampler (stochastic) or the probability-flow ODE (deterministic).

## Evaluation

We evaluate generated samples against a held-out test set using two simple distribution-level metrics:

1. **`mass_w1`**: Wasserstein-1 distance between **per-image mean intensities**  
   $m(x) = \frac{1}{HW}\sum_{i,j} x_{ij}$.
2. **`fft_radial_l2`**: $\ell_2$ distance between the **mean radial power spectra** of real vs generated images (normalised).

Run:

```bash
python scripts/evaluate_sample_quality.py   --real data/toycrystals_test_rotonly_seed1.pt   --gen vae_standard=results/gen_vae_standard.pt   --gen vae_diffusion=results/gen_vae_diff_ddim400.pt   --gen sde=results/gen_sde.pt   --n-real 2048   --n-gen 2048   --csv results/quality_scores.csv
```

You can pass `--gen name=path.pt` **as many times as you like**.

The CSV is written with **semicolon delimiters** (Dutch Excel-friendly).

## Quickstart

Install:

```bash
pip install -e .
```

For the exact commands used to produce the figures and final numbers in this README, see:

- [`docs/reproduce.md`](docs/reproduce.md)

If you only want to *score existing* `.pt` files, the evaluation command above is sufficient.

## Repo structure (high level)

- `scripts/`: training, sampling, evaluation entry points
- `src/toycrystals/`: model and dataset code
- `assets/`: figures used in this README
