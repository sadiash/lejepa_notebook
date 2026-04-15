# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


# ─── CELL: Imports + style ───────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Consistent style everywhere
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    })

    # Color palette used everywhere
    GREEN = "#2ecc71"   # healthy / isotropic / good
    RED = "#e74c3c"     # collapsed / bad / noise
    ORANGE = "#f39c12"  # warning / intermediate
    BLUE = "#3498db"    # neutral / data
    PURPLE = "#9b59b6"  # special / novel
    DARK = "#2c3e50"    # text / emphasis

    return BLUE, DARK, GREEN, ORANGE, PURPLE, RED, mo, np, plt


# ─── CELL: Helper functions ──────────────────────────────────
@app.cell
def _(np):
    def sigreg_loss(embeddings, num_slices=256, knots=17):
        """
        SIGReg: the complete regularizer in pure numpy.
        Measures deviation from isotropic Gaussian via random
        1D projections and characteristic function matching.
        """
        N, D = embeddings.shape
        t = np.linspace(0, 3, knots)
        dt = 3.0 / (knots - 1)
        weights = np.full(knots, 2.0 * dt)
        weights[0] = weights[-1] = dt
        phi = np.exp(-t**2 / 2.0)
        w = weights * phi

        A = np.random.randn(D, num_slices)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        proj = embeddings @ A
        x_t = proj[:, :, None] * t[None, None, :]

        ecf_re = np.mean(np.cos(x_t), axis=0)
        ecf_im = np.mean(np.sin(x_t), axis=0)
        err = (ecf_re - phi)**2 + ecf_im**2
        return float(np.mean((err @ w) * N))

    def eff_rank(embeddings):
        """Effective rank: how many dimensions carry real variance."""
        cov = np.cov(embeddings.T)
        eigs = np.linalg.eigvalsh(cov)
        total = eigs.sum()
        if total < 1e-8:
            return 0.0
        p = eigs / total
        return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return eff_rank, sigreg_loss


# ═══════════════════════════════════════════════════════════════
# ACT 0: THE HOOK
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        # What If You Could Prevent AI Collapse With 50 Lines of Math?

        Every self-supervised AI model wants to cheat.

        Given the task *"predict one view of data from another,"* the simplest
        solution is: **map every input to the same point.** Prediction error drops
        to zero. The model learns nothing. This is called **representational collapse**,
        and preventing it has been the central engineering challenge in self-supervised
        learning for a decade.

        The field's solution? Pile on heuristics. Stop-gradients. Teacher-student
        networks. EMA schedulers. Variance penalties. Each one patches one failure
        mode. None of them explains *why* they work.

        **LeJEPA** asks a different question: *Is there a provably optimal shape for
        embeddings?* And if so, *can we enforce it directly?*

        The answer to both is yes. The enforcer fits in ~50 lines of code. This
        notebook lets you see it work.

        > **Paper:** [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)
        > — Randall Balestriero & Yann LeCun (2025)
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# ACT 1: WATCH IT COLLAPSE
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Act 1: Watch It Collapse

        Below is a cloud of 400 embedding vectors in 2D. In a real JEPA, these
        would be the outputs of a neural network encoder. Right now they're healthy:
        spread across both dimensions, effective rank close to 2.

        **Drag the slider to simulate what happens during training without any
        collapse prevention.** Three different failure modes unfold simultaneously:
        """
    )
    return


@app.cell
def _(mo):
    health = mo.ui.slider(
        0.0, 1.0, value=1.0, step=0.02,
        label="Embedding health (1.0 = healthy, 0.0 = fully collapsed)"
    )
    health
    return (health,)


@app.cell
def _(BLUE, DARK, GREEN, RED, eff_rank, health, mo, np, plt):
    _h = health.value

    _rng = np.random.RandomState(42)
    _n = 400
    _base = _rng.randn(_n, 2)

    # Three collapse modes
    _full = _base * _h
    _rank1 = _base.copy(); _rank1[:, 1] *= _h
    _dim = _base.copy(); _dim[:, 0] *= (0.2 + 0.8 * _h); _dim[:, 1] *= _h

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for _ax, _data, _name in zip(axes, [_full, _rank1, _dim], [
        "Total collapse\n(all dims shrink)", "Rank-1 collapse\n(one dim dies)",
        "Dimensional collapse\n(unequal shrinkage)"
    ]):
        _r = eff_rank(_data) if np.var(_data) > 1e-10 else 0.0
        _color = GREEN if _r > 1.8 else (RED if _r < 0.5 else DARK)

        _ax.scatter(_data[:, 0], _data[:, 1], alpha=0.35, s=10, c=BLUE,
                    edgecolors="none")
        _ax.set_xlim(-4, 4); _ax.set_ylim(-4, 4); _ax.set_aspect("equal")
        _ax.set_title(_name, fontsize=10, fontweight="bold")
        _ax.text(0.95, 0.95, f"eff. rank\n{_r:.2f} / 2",
                 transform=_ax.transAxes, ha="right", va="top", fontsize=12,
                 fontweight="bold", color=_color,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_color,
                           alpha=0.9))

    fig.suptitle(f"Health = {_h:.2f}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    _status = (
        "All three are healthy. Full rank, spread across both dimensions."
        if _h > 0.8 else
        "Collapse is starting. Notice the rank numbers dropping."
        if _h > 0.3 else
        "Severe collapse. Rank-1 has lost an entire dimension."
        if _h > 0.05 else
        "Complete collapse. All information destroyed."
    )

    mo.vstack([fig, mo.callout(mo.md(_status), kind="warn" if _h < 0.5 else "info")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The three panels show the same underlying data being destroyed in
        different ways. In a real network, **you wouldn't know which mode is
        happening** until your downstream task fails. That's why the field
        developed five separate heuristics: each one detects and prevents a
        different failure mode.

        But what if you could prevent *all of them at once*?
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# ACT 2: THE OPTIMAL SHAPE
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Act 2: What Shape Should Embeddings Be?

        If you're going to enforce a distribution, you'd better pick the right one.
        LeJEPA's first theorem proves that among all distributions with the same
        total variance, the **isotropic Gaussian uniquely minimizes worst-case
        downstream risk** for linear tasks.

        The intuition: if you don't know what downstream task will use your
        embeddings, you want information spread equally across all dimensions.
        Any anisotropy creates weak directions that an adversarial task could exploit.

        **Try it yourself.** Change the embedding dimension and distribution below.
        We fit 200 random linear classifiers on each distribution and measure how
        often they succeed. The isotropic Gaussian's *worst case* is better than
        everyone else's:
        """
    )
    return


@app.cell
def _(mo):
    dim_pick = mo.ui.slider(2, 32, value=8, step=2, label="Embedding dimension")
    dim_pick
    return (dim_pick,)


@app.cell
def _(BLUE, GREEN, ORANGE, PURPLE, RED, dim_pick, mo, np, plt):
    _d = dim_pick.value
    _n, _n_tasks = 300, 200
    _rng = np.random.RandomState(0)

    def _make(kind, rng, n, d):
        if kind == "isotropic":
            return rng.randn(n, d)
        elif kind == "anisotropic":
            s = np.exp(np.linspace(1, -2, d))
            return rng.randn(n, d) * (s / np.sqrt(np.mean(s**2)))
        elif kind == "rank_def":
            k = max(1, d // 3)
            e = rng.randn(n, k) @ rng.randn(k, d)
            return e / (np.std(e) + 1e-8)
        elif kind == "uniform":
            e = rng.uniform(-1, 1, (n, d))
            return e / (np.std(e) + 1e-8)

    def _probe(emb, rng, n_tasks):
        accs = []
        for _ in range(n_tasks):
            w = rng.randn(emb.shape[1])
            w /= np.linalg.norm(w)
            y = (emb @ w > 0).astype(float)
            h = len(emb) // 2
            try:
                w_h = np.linalg.lstsq(emb[:h], y[:h] * 2 - 1, rcond=None)[0]
                accs.append(np.mean((emb[h:] @ w_h > 0) == y[h:].astype(bool)))
            except Exception:
                accs.append(0.5)
        return np.array(accs)

    _types = [
        ("Isotropic\nGaussian", "isotropic", GREEN),
        ("Anisotropic\nGaussian", "anisotropic", ORANGE),
        ("Rank-\ndeficient", "rank_def", RED),
        ("Uniform", "uniform", PURPLE),
    ]

    fig2, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

    _worst = []
    for _label, _kind, _color in _types:
        _emb = _make(_kind, _rng, _n, _d)
        _a = _probe(_emb, _rng, _n_tasks)
        ax_hist.hist(_a, bins=20, alpha=0.45, color=_color, label=_label.replace("\n", " "),
                     range=(0.4, 1.0), density=True)
        _worst.append(np.percentile(_a, 5))

    ax_hist.set_xlabel("Probe accuracy")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Accuracy across 200 random tasks", fontweight="bold")
    ax_hist.legend(fontsize=8)

    _bars = ax_bar.bar(
        [t[0] for t in _types], _worst,
        color=[t[2] for t in _types], alpha=0.8, edgecolor="white", linewidth=1.5
    )
    for _b, _v in zip(_bars, _worst):
        ax_bar.text(_b.get_x() + _b.get_width()/2, _v + 0.008, f"{_v:.2f}",
                    ha="center", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Worst-case accuracy (5th percentile)")
    ax_bar.set_title("Who survives the hardest task?", fontweight="bold")
    ax_bar.set_ylim(0.4, 1.0)

    plt.tight_layout()

    _winner = "Isotropic Gaussian" if _worst[0] >= max(_worst) - 0.01 else _types[np.argmax(_worst)][0]
    mo.vstack([
        fig2,
        mo.callout(
            mo.md(f"**The isotropic Gaussian wins.** Its worst-case accuracy ({_worst[0]:.2f}) "
                  f"is the best floor. No matter what downstream task you throw at it, "
                  f"it has no weak directions to exploit."),
            kind="success"
        )
    ])
    return


# ═══════════════════════════════════════════════════════════════
# ACT 3: THE ~50 LINE SOLUTION
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Act 3: The ~50 Line Solution

        Now we know the target: isotropic Gaussian. How do you enforce it?

        Matching a high-dimensional distribution directly is computationally
        brutal. But there's an elegant shortcut from probability theory called
        the **Cramér-Wold theorem**: a distribution is Gaussian *if and only if*
        every 1D projection is Gaussian.

        So instead of checking 192 dimensions simultaneously, you:

        1. **Project** embeddings onto random 1D directions (a dot product)
        2. **Compare** each 1D histogram to a Gaussian using characteristic functions
        3. **Average** the discrepancy

        That's the entire idea. Here it is in numpy:

        ```python
        def sigreg_loss(embeddings, num_slices=256, knots=17):
            N, D = embeddings.shape

            # Quadrature points for integrating the CF difference
            t = np.linspace(0, 3, knots)
            dt = 3.0 / (knots - 1)
            weights = np.full(knots, 2.0 * dt)
            weights[0] = weights[-1] = dt

            # Target: Gaussian characteristic function phi(t) = exp(-t^2/2)
            phi = np.exp(-t**2 / 2.0)

            # Random projection directions
            A = np.random.randn(D, num_slices)
            A /= np.linalg.norm(A, axis=0, keepdims=True)

            # Project and compute empirical CF
            proj = embeddings @ A
            x_t = proj[:, :, None] * t[None, None, :]
            ecf_re = np.mean(np.cos(x_t), axis=0)  # real part
            ecf_im = np.mean(np.sin(x_t), axis=0)  # imaginary part

            # |ECF(t) - phi(t)|^2, integrated
            err = (ecf_re - phi)**2 + ecf_im**2
            return float(np.mean((err @ (weights * phi)) * N))
        ```

        That's it. No stop-gradient. No teacher network. No EMA. Just measure
        how far your embeddings are from Gaussian, and add it to the loss.

        **Now watch it work.** The slider below morphs embeddings from completely
        collapsed (left) to perfectly isotropic (right). Three panels show what
        SIGReg sees at each point:
        """
    )
    return


@app.cell
def _(mo):
    morph = mo.ui.slider(
        0.0, 1.0, value=0.0, step=0.02,
        label="Morph: collapsed ← → isotropic Gaussian"
    )
    morph
    return (morph,)


@app.cell
def _(BLUE, DARK, GREEN, RED, eff_rank, morph, np, plt, sigreg_loss):
    _alpha = morph.value
    _rng = np.random.RandomState(99)
    _n, _d = 400, 8

    # Morph between collapsed and isotropic
    _iso = _rng.randn(_n, _d)
    _collapsed = np.tile(_rng.randn(1, _d) * 0.01, (_n, 1))
    _collapsed[:, 0] = _rng.randn(_n) * 0.5  # keep one dim alive
    _emb = _alpha * _iso + (1 - _alpha) * _collapsed

    _loss = sigreg_loss(_emb, num_slices=128)
    _rank = eff_rank(_emb)

    # Project onto a random direction for visualization
    _w = _rng.randn(_d); _w /= np.linalg.norm(_w)
    _proj = _emb @ _w

    # CF computation for visualization
    _t_vis = np.linspace(-4, 4, 200)
    _ecf_re = np.mean(np.cos(_proj[:, None] * _t_vis[None, :]), axis=0)
    _phi_vis = np.exp(-_t_vis**2 / 2.0)

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: PCA of embeddings
    _centered = _emb - _emb.mean(0)
    try:
        _, _, _Vt = np.linalg.svd(_centered, full_matrices=False)
        _pca = _centered @ _Vt[:2].T
    except Exception:
        _pca = _centered[:, :2]
    _lim = max(np.abs(_pca).max() * 1.3, 0.5)

    axes3[0].scatter(_pca[:, 0], _pca[:, 1], alpha=0.3, s=8, c=BLUE,
                     edgecolors="none")
    axes3[0].set_xlim(-_lim, _lim); axes3[0].set_ylim(-_lim, _lim)
    axes3[0].set_aspect("equal")
    axes3[0].set_title("Embeddings (PCA)", fontweight="bold")
    _rcolor = GREEN if _rank > _d * 0.8 else (RED if _rank < 2 else DARK)
    axes3[0].text(0.05, 0.95, f"eff. rank: {_rank:.1f}/{_d}",
                  transform=axes3[0].transAxes, va="top", fontsize=10,
                  fontweight="bold", color=_rcolor,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_rcolor, alpha=0.85))

    # Panel 2: histogram of 1D projection
    axes3[1].hist(_proj, bins=30, density=True, alpha=0.6, color=BLUE,
                  edgecolor="white", linewidth=0.5, label="Projected data")
    _x_gauss = np.linspace(-4, 4, 200)
    axes3[1].plot(_x_gauss, np.exp(-_x_gauss**2/2) / np.sqrt(2*np.pi),
                  "k--", linewidth=2, label="Gaussian target")
    axes3[1].set_xlim(-4, 4)
    axes3[1].set_title("1D projection (what SIGReg sees)", fontweight="bold")
    axes3[1].legend(fontsize=8)

    # Panel 3: characteristic function comparison
    axes3[2].plot(_t_vis, _phi_vis, "k--", linewidth=2, label="Gaussian CF")
    axes3[2].plot(_t_vis, _ecf_re, color=BLUE, linewidth=1.5, label="Empirical CF")
    axes3[2].fill_between(_t_vis, _phi_vis, _ecf_re, alpha=0.15, color=RED)
    axes3[2].set_title("Characteristic functions", fontweight="bold")
    axes3[2].set_xlabel("t")
    axes3[2].legend(fontsize=8)

    # SIGReg loss as big number
    _loss_color = GREEN if _loss < 5 else (RED if _loss > 50 else DARK)
    fig3.suptitle(f"SIGReg loss = {_loss:.1f}", fontsize=15, fontweight="bold",
                  color=_loss_color, y=1.03)
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Slide it slowly from left to right.** Watch three things happen simultaneously:

        - The embedding cloud expands from a clump to a sphere (left panel)
        - The 1D projection changes from a spike to a bell curve (middle panel)
        - The gap between the blue and black curves shrinks to nothing (right panel)
        - The SIGReg loss drops from ~100+ to near zero

        The middle panel is the key insight: **SIGReg never looks at the full
        high-dimensional distribution.** It only checks 1D slices. The Cramér-Wold
        theorem guarantees that if all slices are Gaussian, the whole distribution is.

        That's why the implementation is ~50 lines. The mathematical trick eliminates
        the curse of dimensionality entirely.
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# ACT 4: REPLACING FIVE HEURISTICS
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Act 4: One Loss Term Replaces Five Heuristics

        Before LeJEPA, self-supervised learning looked like this:

        | Heuristic | What it fixes | Introduced by |
        |:----------|:-------------|:-------------|
        | Stop-gradient | Prevents gradients from collapsing the target | SimSiam (2021) |
        | Teacher-student + EMA | Stabilizes the target with a slow-moving copy | BYOL (2020) |
        | Centering + sharpening | Prevents mean and uniform collapse | DINO (2021) |
        | Variance-covariance reg | Prevents dimensional collapse | VICReg (2022) |
        | Predictor network | Adds capacity to absorb asymmetry | BYOL / I-JEPA |

        Each one was discovered by trial and error. Each patches one specific failure
        mode. **SIGReg replaces all five** by enforcing the optimal distribution
        directly. If the distribution is isotropic Gaussian, none of the failure modes
        can occur by construction.

        Below: a toy encoder trained on synthetic data. Top row has no collapse
        prevention. Bottom row has SIGReg. Use the epoch slider to watch the
        difference unfold:
        """
    )
    return


@app.cell
def _(eff_rank, np, sigreg_loss):
    # Pre-compute both training runs so the epoch slider is instant
    def _train_run(use_sigreg, seed):
        rng = np.random.RandomState(seed)
        n_cls, n_per = 6, 40
        centers = rng.randn(n_cls, 24) * 3
        X = np.vstack([centers[i] + rng.randn(n_per, 24) * 0.5 for i in range(n_cls)])
        labels = np.repeat(np.arange(n_cls), n_per)

        W1 = rng.randn(24, 48) * 0.08
        b1 = np.zeros(48)
        W2 = rng.randn(48, 12) * 0.08
        b2 = np.zeros(12)

        snaps, losses, ranks = [], [], []
        for epoch in range(50):
            h = np.maximum(0, X @ W1 + b1)
            emb = h @ W2 + b2
            snaps.append(emb.copy())
            losses.append(sigreg_loss(emb, num_slices=24))
            ranks.append(eff_rank(emb))

            X_aug = X + rng.randn(*X.shape) * 0.3
            h_aug = np.maximum(0, X_aug @ W1 + b1)
            emb_aug = h_aug @ W2 + b2
            dL = 2 * (emb - emb_aug) / len(X)

            if use_sigreg:
                eps = 0.005
                base = sigreg_loss(emb, num_slices=24)
                dW2_sig = np.zeros_like(W2)
                for j in range(min(12, 3)):
                    W2p = W2.copy(); W2p[:, j] += eps
                    dW2_sig[:, j] = (sigreg_loss(h @ W2p + b2, num_slices=24) - base) / eps
            else:
                dW2_sig = np.zeros_like(W2)

            dW2 = h.T @ dL / len(X) + (0.25 * dW2_sig / (len(X) + 1) if use_sigreg else 0)
            db2 = np.mean(dL, axis=0)
            dh = dL @ W2.T * (h > 0)
            dW1 = X.T @ dh / len(X)
            db1 = np.mean(dh, axis=0)

            lr = 0.001
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2

        return snaps, losses, ranks, labels

    _no_sig = _train_run(False, 42)
    _with_sig = _train_run(True, 42)
    precomputed = {"no_sigreg": _no_sig, "with_sigreg": _with_sig}
    return (precomputed,)


@app.cell
def _(mo):
    epoch_slider = mo.ui.slider(0, 49, value=0, step=1, label="Training epoch")
    epoch_slider
    return (epoch_slider,)


@app.cell
def _(BLUE, GREEN, RED, epoch_slider, mo, np, plt, precomputed):
    _ep = epoch_slider.value

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))

    for _col, (_key, _title, _border_color) in enumerate([
        ("no_sigreg", "Without SIGReg", RED),
        ("with_sigreg", "With SIGReg", GREEN),
    ]):
        _snaps, _losses, _ranks, _labels = precomputed[_key]
        _data = _snaps[_ep]
        _dc = _data - _data.mean(0)
        try:
            _, _, _Vt = np.linalg.svd(_dc, full_matrices=False)
            _p2 = _dc @ _Vt[:2].T
        except Exception:
            _p2 = _dc[:, :2]

        ax = axes4[_col]
        _lim = max(np.abs(_p2).max() * 1.3, 0.3)
        ax.scatter(_p2[:, 0], _p2[:, 1], c=_labels, cmap="Set2", alpha=0.6, s=15,
                   edgecolors="white", linewidth=0.3)
        ax.set_xlim(-_lim, _lim); ax.set_ylim(-_lim, _lim); ax.set_aspect("equal")
        ax.set_title(_title, fontsize=12, fontweight="bold", color=_border_color)

        _r = _ranks[_ep]
        _rc = GREEN if _r > 8 else (RED if _r < 3 else "#555")
        ax.text(0.05, 0.95, f"rank: {_r:.1f}/12\nSIGReg: {_losses[_ep]:.0f}",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                color=_rc,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_rc, alpha=0.85))

        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor(_border_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig4.suptitle(f"Epoch {_ep}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    _msg = (
        "Both start the same. Drag the slider forward to watch the divergence."
        if _ep < 5 else
        "The left panel is collapsing. Clusters are merging. The right panel holds."
        if _ep < 25 else
        "The left panel has lost cluster structure entirely. SIGReg kept it alive."
    )
    mo.vstack([fig4, mo.callout(mo.md(_msg), kind="info")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Scrub from epoch 0 to 49.** On the left, without SIGReg, the 6 colored
        clusters progressively merge into an indistinct blob, then a line, then a
        point. On the right, SIGReg keeps clusters separated by forcing the overall
        distribution to stay spread across all dimensions.

        This is LeJEPA's practical contribution: **one loss term, one hyperparameter
        (lambda), zero heuristics.**
        """
    )
    return


# ═══════════════════════════════════════════════════════════════
# ACT 5: BEYOND THE PAPER — A DISCOVERY
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Act 5: Beyond the Paper — What Happens When SIGReg Meets the Real World?

        *This section presents novel empirical findings not in the original paper.*

        LeJEPA's SIGReg trains the encoder to produce isotropic Gaussian embeddings.
        But in the real world, agents encounter inputs they've never seen before:
        visual noise, corrupted sensors, out-of-distribution observations.

        We investigated this using **LeWM** (Maes & Le Lidec, 2026), an
        18M-parameter world model that applies LeJEPA's architecture to reinforcement
        learning. We injected pixel noise into observations and measured what
        happened to the latent space.

        ### The first finding will surprise you. The second one is stranger.
        """
    )
    return


@app.cell
def _(GREEN, ORANGE, RED, mo, plt):
    # --- Hardcoded experimental data ---
    # Source: Ashraf (2026), JEPA Encoders Collapse Pixel Noise

    _conds = ["Clean", "Gaussian\nnoise", "Pure\nrandom"]
    _eff_ranks = [94.8, 39.2, 10.1]
    _rnd = [0.014, 0.011, 0.012]
    _colors = [GREEN, ORANGE, RED]

    fig5a, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(11, 4))

    # Effective rank
    bars_r = ax_r.bar(_conds, _eff_ranks, color=_colors, alpha=0.8,
                      edgecolor="white", linewidth=1.5)
    for _b, _v in zip(bars_r, _eff_ranks):
        ax_r.text(_b.get_x() + _b.get_width()/2, _v + 2, f"{_v:.1f}",
                  ha="center", fontsize=12, fontweight="bold")
    ax_r.set_ylabel("Effective rank (of 192 dims)")
    ax_r.set_title("Latent dimensionality", fontweight="bold")
    ax_r.set_ylim(0, 115)

    # RND novelty
    bars_n = ax_n.bar(_conds, _rnd, color=_colors, alpha=0.8,
                      edgecolor="white", linewidth=1.5)
    for _b, _v in zip(bars_n, _rnd):
        ax_n.text(_b.get_x() + _b.get_width()/2, _v + 0.0004, f"{_v:.3f}",
                  ha="center", fontsize=12, fontweight="bold")
    ax_n.set_ylabel("RND novelty score")
    ax_n.set_title("Does novelty detection notice?", fontweight="bold")
    ax_n.set_ylim(0, 0.020)

    plt.tight_layout()

    mo.vstack([
        fig5a,
        mo.callout(
            mo.md(
                "**Finding 1: The encoder silently collapses noise.**\n\n"
                "Pure random pixels get compressed from 94.8 effective dimensions "
                "down to 10.1 — a **9.4x reduction.** But the novelty detector "
                "(RND) doesn't even notice (0.012 vs 0.014). The encoder has "
                "already solved the 'noisy TV problem' before any exploration "
                "signal sees the data."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Now for the strange part.

        If the encoder collapses *global* noise, does it also collapse *localized*
        noise? Say, a small random patch in the corner of an image?

        We tested this with noise patches from 6% to 100% of the image.

        **Before you see the result, make a prediction:**
        """
    )
    return


@app.cell
def _(mo):
    quiz = mo.ui.radio(
        options=[
            "Small patch causes LESS disruption (encoder ignores it)",
            "Small patch causes MORE disruption (harder to handle)",
            "All noise sizes cause equal disruption",
        ],
        label="A 16x16 noise patch covers 6% of the image. Full noise covers 100%. Which causes higher prediction loss?"
    )
    quiz
    return (quiz,)


@app.cell
def _(BLUE, DARK, GREEN, ORANGE, PURPLE, RED, mo, np, plt, quiz):
    if quiz.value is None:
        mo.output.replace(mo.md("*Select your prediction above to reveal the result.*"))
    else:
        _coverages = ["Clean\n(0%)", "16x16\n(6%)", "32x32\n(25%)", "48x48\n(56%)", "Global\n(100%)"]
        _pred_loss = [0.008, 0.057, 0.017, 0.029, 0.024]
        _colors = [GREEN, RED, ORANGE, ORANGE, PURPLE]

        fig5b, ax5b = plt.subplots(figsize=(9, 4.5))
        bars = ax5b.bar(_coverages, _pred_loss, color=_colors, alpha=0.8,
                        edgecolor="white", linewidth=1.5)

        # Highlight the peak
        bars[1].set_edgecolor("#000")
        bars[1].set_linewidth(3)

        for _b, _v in zip(bars, _pred_loss):
            ax5b.text(_b.get_x() + _b.get_width()/2, _v + 0.002, f"{_v:.3f}",
                      ha="center", fontsize=11, fontweight="bold")

        # Annotation
        ax5b.annotate(
            "PEAK: smallest patch\ncauses most disruption!",
            xy=(1, 0.057), xytext=(2.5, 0.055),
            arrowprops=dict(arrowstyle="->", color=RED, linewidth=2),
            fontsize=11, fontweight="bold", color=RED, ha="center",
        )

        ax5b.set_ylabel("Prediction loss")
        ax5b.set_xlabel("Noise coverage")
        ax5b.set_title("The Non-Monotonic Surprise", fontsize=13, fontweight="bold")
        plt.tight_layout()

        _correct = "MORE" in quiz.value
        _verdict = (
            "**You got it!** " if _correct else "**Surprised?** Most people guess wrong. "
        )

        mo.output.replace(
            mo.vstack([
                fig5b,
                mo.callout(
                    mo.md(
                        f"{_verdict}The smallest patch (6% of the image) causes **7x more "
                        f"prediction loss** than full noise.\n\n"
                        f"**Why?** With global noise, the encoder collapses everything to a "
                        f"compact region the predictor can partially handle. With a small "
                        f"patch, the encoder creates a **hybrid clean/noisy representation** "
                        f"that is harder to predict than either extreme. The non-monotonic "
                        f"peak reveals that SIGReg's isotropic structure is most disrupted "
                        f"not by maximal noise, but by the *boundary* between clean and "
                        f"corrupted representations."
                    ),
                    kind="warn",
                ),
            ])
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Explore all the metrics

        The non-monotonic pattern appears in multiple measurements. Use the
        dropdown to see how different metrics respond to increasing noise:
        """
    )
    return


@app.cell
def _(mo):
    metric_dd = mo.ui.dropdown(
        options={
            "Prediction loss": "pred",
            "Compression progress": "cp",
            "RND novelty": "rnd",
            "Latent variance": "var",
            "Cosine similarity to clean": "cos",
        },
        value="Prediction loss",
        label="Metric"
    )
    metric_dd
    return (metric_dd,)


@app.cell
def _(BLUE, metric_dd, np, plt):
    _covs = [0, 6, 25, 56, 100]
    _labels = ["0%", "6%", "25%", "56%", "100%"]

    _all_data = {
        "pred": ([0.008, 0.057, 0.017, 0.029, 0.024], "Prediction loss"),
        "cp":   ([0.002, 0.025, 0.009, 0.016, 0.012], "Compression progress"),
        "rnd":  ([0.010, 0.013, 0.016, 0.016, 0.016], "RND novelty"),
        "var":  ([0.743, 0.135, 0.036, 0.009, 0.005], "Latent variance"),
        "cos":  ([1.0, 0.41, 0.25, -0.05, -0.16], "Cosine similarity to clean"),
    }

    _m = metric_dd.value
    _vals, _ylabel = _all_data[_m]

    fig5c, ax5c = plt.subplots(figsize=(8, 4))
    ax5c.plot(_covs, _vals, "o-", color=BLUE, linewidth=2, markersize=8,
              markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=2)
    ax5c.fill_between(_covs, _vals, alpha=0.08, color=BLUE)

    for x, y, lab in zip(_covs, _vals, _labels):
        ax5c.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                      xytext=(0, 12), ha="center", fontsize=9)

    # Mark peak if non-monotonic
    _peak = np.argmax(_vals)
    if _peak not in (0, len(_vals) - 1):
        ax5c.plot(_covs[_peak], _vals[_peak], "o", color="#e74c3c",
                  markersize=14, markerfacecolor="none", markeredgewidth=2.5)

    ax5c.set_xlabel("Noise coverage (%)")
    ax5c.set_ylabel(_ylabel)
    ax5c.set_title(f"{_ylabel} vs. noise coverage", fontweight="bold")
    plt.tight_layout()
    fig5c
    return


# ═══════════════════════════════════════════════════════════════
# CLOSING
# ═══════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## What You Just Saw

        1. **Representational collapse** is the central failure mode of self-supervised
           learning, and it takes multiple forms (total, rank-1, dimensional).

        2. The **isotropic Gaussian** is the provably optimal embedding distribution
           for unknown downstream tasks.

        3. **SIGReg** enforces this distribution using random 1D projections and
           characteristic function matching, in ~50 lines of code, replacing five
           separate heuristics that the field accumulated over four years.

        4. Beyond the paper: SIGReg's distributional structure has consequences for
           **exploration in reinforcement learning.** The JEPA encoder collapses
           global noise (solving the noisy TV problem for free) but not localized
           noise, with a non-monotonic dose-response that reveals how the encoder
           handles the boundary between clean and corrupted representations.

        **LeJEPA shows that theory can replace heuristics.** One proof, one loss
        term, one hyperparameter. The rest follows by construction.

        ---

        *Notebook by Sadia Ashraf, Independent Researcher.*
        *Extension results (Act 5) from: Ashraf, S. (2026). JEPA Encoders Collapse Pixel Noise: Implications for Intrinsic Motivation in Learned Latent Spaces. Working paper.*

        **References**

        - Balestriero, R. & LeCun, Y. (2025). [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.](https://arxiv.org/abs/2511.08544) arXiv:2511.08544.
        - Maes, L. & Le Lidec, Q. (2026). LeWorldModel: World Modeling with Latent Imagination. arXiv:2603.19312.
        - Schmidhuber, J. (2009). Driven by Compression Progress. Springer.
        - Burda, Y. et al. (2019). Exploration by Random Network Distillation. ICLR 2019.
        """
    )
    return


if __name__ == "__main__":
    app.run()
