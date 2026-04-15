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


# ============================================================
# CELL: Imports
# ============================================================
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    np.random.seed(42)
    return GridSpec, mo, mcolors, np, plt


# ============================================================
# CELL: Title
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        # Bringing LeJEPA to Life

        **Paper:** [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)
        — Randall Balestriero & Yann LeCun (2025)

        **Notebook by:** Sadia Ashraf, Independent Researcher

        ---

        Joint-Embedding Predictive Architectures (JEPAs) learn by predicting one view's embedding from another.
        The open question has always been: **what stops the embeddings from collapsing to a trivial constant?**

        Previous solutions piled on heuristics: stop-gradients, teacher-student networks, EMA schedulers.
        LeJEPA replaces all of them with a single theoretical insight and a regularizer that fits in ~50 lines of code.

        This notebook walks through the core contribution in five sections:

        1. **The collapse problem** — what goes wrong without regularization
        2. **Why isotropic Gaussian is optimal** — the theoretical result
        3. **SIGReg in ~50 lines** — the regularizer, implemented live
        4. **Old heuristics vs. SIGReg** — why they're all redundant now
        5. **Extension: SIGReg meets intrinsic motivation** — novel empirical findings connecting LeJEPA to exploration in world models
        """
    )
    return


# ============================================================
# SECTION 1: The Collapse Problem
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. The Collapse Problem, Visualized

        A JEPA trains two networks: an encoder maps inputs to embeddings, and a predictor
        maps one embedding to another. The training objective minimizes prediction error
        between the two embeddings.

        The problem: if both embeddings collapse to the same constant vector, prediction
        error is zero. **The network "solves" the task by learning nothing.**

        Collapse can be total (all embeddings map to one point), partial (embeddings
        occupy a low-rank subspace), or dimensional (some dimensions carry signal,
        others are dead). All three are failure modes that previous JEPAs fought with
        heuristics.

        Use the slider below to see what collapse looks like in a 2D embedding space.
        """
    )
    return


@app.cell
def _(mo):
    collapse_slider = mo.ui.slider(
        0.0, 1.0, value=1.0, step=0.05,
        label="Embedding health (1.0 = healthy, 0.0 = fully collapsed)"
    )
    mo.md(f"**Adjust collapse severity:** {collapse_slider}")
    return (collapse_slider,)


@app.cell
def _(collapse_slider, np, plt):
    _health = collapse_slider.value

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 3.5))

    # Generate healthy embeddings (isotropic Gaussian)
    _n = 400
    _healthy = np.random.randn(_n, 2)

    # Full collapse: all points converge to origin
    _full = _healthy * _health + (1 - _health) * np.array([0.0, 0.0])

    # Rank-1 collapse: second dimension dies
    _rank1 = _healthy.copy()
    _rank1[:, 1] = _healthy[:, 1] * _health

    # Dimensional collapse: variance becomes anisotropic
    _dim = _healthy.copy()
    _dim[:, 0] = _healthy[:, 0] * (0.2 + 0.8 * _health)
    _dim[:, 1] = _healthy[:, 1] * _health

    for _ax, _data, _title in zip(
        axes1,
        [_full, _rank1, _dim],
        ["Full collapse", "Rank-1 collapse", "Dimensional collapse"],
    ):
        _ax.scatter(_data[:, 0], _data[:, 1], alpha=0.3, s=8, c="#4a90d9")
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect("equal")
        _ax.set_title(_title, fontsize=11)
        _ax.axhline(0, color="#ccc", linewidth=0.5, zorder=0)
        _ax.axvline(0, color="#ccc", linewidth=0.5, zorder=0)

        # Show effective rank
        _cov = np.cov(_data.T)
        _eigvals = np.linalg.eigvalsh(_cov)
        _total_var = _eigvals.sum()
        if _total_var < 1e-8:
            _eff_rank = 0.0
        else:
            _p = _eigvals / _total_var
            _eff_rank = np.exp(-np.sum(_p * np.log(_p + 1e-12)))
        _ax.text(
            0.05, 0.95, f"eff. rank: {_eff_rank:.2f}/2",
            transform=_ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig1.suptitle(
        f"Health = {_health:.2f}", fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig1
    return


# ============================================================
# SECTION 2: Why Isotropic Gaussian?
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Why Isotropic Gaussian Is Optimal

        LeJEPA's first theoretical contribution (Theorem 1 in the paper): among all
        embedding distributions with the same total variance, the **isotropic Gaussian
        uniquely minimizes the worst-case risk over all possible downstream linear tasks**.

        The intuition: if you don't know what downstream task your embeddings will be
        used for, the safest bet is to spread information equally across all dimensions.
        Any anisotropy means some directions carry less information, and an adversarial
        downstream task could exploit exactly those weak directions.

        The visualization below samples embeddings from four different distributions,
        all with the same total variance. For each, we fit linear probes on 200 random
        binary classification tasks and measure accuracy. **The isotropic Gaussian has
        the best worst-case performance.**
        """
    )
    return


@app.cell
def _(mo):
    dim_slider = mo.ui.slider(
        2, 32, value=8, step=2, label="Embedding dimension"
    )
    n_samples_slider = mo.ui.slider(
        50, 500, value=200, step=50, label="Number of samples"
    )
    mo.md(
        f"**Parameters:** {dim_slider} &nbsp;&nbsp; {n_samples_slider}"
    )
    return dim_slider, n_samples_slider


@app.cell
def _(dim_slider, n_samples_slider, np, plt):
    _d = dim_slider.value
    _n = n_samples_slider.value
    _n_tasks = 200

    def _make_embeddings(dist_type, n, d):
        if dist_type == "Isotropic Gaussian":
            return np.random.randn(n, d)
        elif dist_type == "Anisotropic Gaussian":
            scales = np.exp(np.linspace(1, -2, d))
            scales = scales / np.sqrt(np.mean(scales**2))  # normalize variance
            return np.random.randn(n, d) * scales
        elif dist_type == "Rank-deficient":
            k = max(1, d // 3)
            z = np.random.randn(n, k)
            A = np.random.randn(k, d)
            emb = z @ A
            emb = emb / np.std(emb) * 1.0
            return emb
        elif dist_type == "Uniform hypercube":
            emb = np.random.uniform(-1, 1, (n, d))
            emb = emb / np.std(emb) * 1.0
            return emb

    def _eval_linear_probes(emb, n_tasks):
        """Fit random linear binary classifiers and return accuracy distribution."""
        accuracies = []
        for _ in range(n_tasks):
            # Random linear task: sign(w^T x)
            w = np.random.randn(emb.shape[1])
            w = w / np.linalg.norm(w)
            true_labels = (emb @ w > 0).astype(int)

            # Train a linear probe (logistic-like: just use the optimal linear separator)
            # Split into train/test
            half = len(emb) // 2
            X_tr, y_tr = emb[:half], true_labels[:half]
            X_te, y_te = emb[half:], true_labels[half:]

            # Least-squares probe
            try:
                w_hat = np.linalg.lstsq(X_tr, y_tr * 2 - 1, rcond=None)[0]
                preds = (X_te @ w_hat > 0).astype(int)
                acc = np.mean(preds == y_te)
                accuracies.append(acc)
            except Exception:
                accuracies.append(0.5)
        return np.array(accuracies)

    _dist_types = [
        "Isotropic Gaussian",
        "Anisotropic Gaussian",
        "Rank-deficient",
        "Uniform hypercube",
    ]
    _colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    _results = {}
    for _dtype, _color in zip(_dist_types, _colors):
        _emb = _make_embeddings(_dtype, _n, _d)
        _accs = _eval_linear_probes(_emb, _n_tasks)
        _results[_dtype] = _accs
        axes2[0].hist(
            _accs, bins=20, alpha=0.5, label=_dtype, color=_color,
            range=(0.4, 1.0), density=True
        )

    axes2[0].set_xlabel("Probe accuracy")
    axes2[0].set_ylabel("Density")
    axes2[0].set_title("Accuracy distribution over random tasks")
    axes2[0].legend(fontsize=8)

    # Worst-case (5th percentile) comparison
    _names = []
    _worst_cases = []
    for _dtype, _color in zip(_dist_types, _colors):
        _names.append(_dtype.replace(" ", "\n"))
        _worst_cases.append(np.percentile(_results[_dtype], 5))

    _bars = axes2[1].bar(_names, _worst_cases, color=_colors, alpha=0.7)
    axes2[1].set_ylabel("5th percentile accuracy\n(worst-case)")
    axes2[1].set_title("Worst-case downstream performance")
    axes2[1].set_ylim(0.4, 1.0)
    for _bar, _val in zip(_bars, _worst_cases):
        axes2[1].text(
            _bar.get_x() + _bar.get_width() / 2, _val + 0.01,
            f"{_val:.2f}", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    fig2
    return


# ============================================================
# SECTION 3: SIGReg in ~50 Lines
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. SIGReg in ~50 Lines, Live

        The paper's second contribution is **SIGReg** (Sketched Isotropic Gaussian
        Regularization), a loss function that pushes embeddings toward the isotropic
        Gaussian distribution.

        The idea uses the [Cramér-Wold theorem](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Wold_theorem):
        a multivariate distribution is Gaussian if and only if **every 1D projection is Gaussian**.
        So instead of matching a high-dimensional distribution (expensive), SIGReg:

        1. Projects embeddings onto random 1D directions (cheap)
        2. Compares each projected distribution to a 1D Gaussian using characteristic functions
        3. Averages the discrepancy across projections

        The characteristic function of a standard Gaussian is $\phi(t) = e^{-t^2/2}$.
        The empirical characteristic function of the projected data is
        $\hat{\phi}(t) = \frac{1}{N}\sum_i e^{it x_i}$. SIGReg measures the
        integrated squared difference between these two.

        Here is the complete implementation in numpy:
        """
    )
    return


@app.cell
def _(np):
    def sigreg_loss(embeddings, num_slices=256, knots=17):
        """
        SIGReg: Sketched Isotropic Gaussian Regularization.

        Measures how far an embedding distribution deviates from
        an isotropic Gaussian, using random 1D projections and
        characteristic function matching.

        Args:
            embeddings: (N, D) array of embedding vectors
            num_slices: number of random projection directions
            knots: quadrature points for CF integration

        Returns:
            Scalar loss (0 = perfect isotropic Gaussian)
        """
        N, D = embeddings.shape

        # --- Quadrature setup (trapezoidal rule on [0, 3]) ---
        t = np.linspace(0, 3, knots)
        dt = 3.0 / (knots - 1)
        weights = np.full(knots, 2.0 * dt)
        weights[0] = weights[-1] = dt
        # Gaussian CF: phi(t) = exp(-t^2 / 2)
        phi_gaussian = np.exp(-t**2 / 2.0)
        weighted = weights * phi_gaussian

        # --- Random projection directions ---
        A = np.random.randn(D, num_slices)
        A = A / np.linalg.norm(A, axis=0, keepdims=True)

        # --- Project: (N, D) @ (D, S) = (N, S) ---
        proj = embeddings @ A

        # --- Empirical CF vs Gaussian CF ---
        # x_t[i, s, k] = proj[i, s] * t[k]
        x_t = proj[:, :, None] * t[None, None, :]

        # ECF = (1/N) * sum_i exp(i * x_i * t)
        ecf_real = np.mean(np.cos(x_t), axis=0)  # (S, knots)
        ecf_imag = np.mean(np.sin(x_t), axis=0)  # (S, knots)

        # Squared difference: |ECF(t) - phi(t)|^2
        err = (ecf_real - phi_gaussian)**2 + ecf_imag**2

        # Integrate over t, scale by N (the statistic)
        statistic = (err @ weighted) * N
        return float(np.mean(statistic))


    def effective_rank(embeddings):
        """Effective rank via eigenvalue entropy."""
        cov = np.cov(embeddings.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        p = eigvals / eigvals.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return effective_rank, sigreg_loss


@app.cell
def _(mo):
    mo.md(
        r"""
        ### SIGReg on toy embeddings

        Below, we generate 2D embeddings from different distributions and compute
        the SIGReg loss for each. The isotropic Gaussian gets the lowest loss.
        Drag the **lambda slider** to control the regularization strength in a
        gradient descent animation that reshapes collapsed embeddings toward the
        isotropic Gaussian target.
        """
    )
    return


@app.cell
def _(mo):
    lam_slider = mo.ui.slider(
        0.0, 1.0, value=0.5, step=0.05,
        label="Lambda (SIGReg weight)"
    )
    steps_slider = mo.ui.slider(
        0, 60, value=30, step=5,
        label="Gradient descent steps"
    )
    mo.md(f"**Controls:** {lam_slider} &nbsp;&nbsp; {steps_slider}")
    return lam_slider, steps_slider


@app.cell
def _(effective_rank, lam_slider, np, plt, sigreg_loss, steps_slider):
    _lam = lam_slider.value
    _n_steps = steps_slider.value

    # Start from collapsed embeddings (rank-1)
    _rng = np.random.RandomState(7)
    _n = 300
    _emb = _rng.randn(_n, 2)
    _emb[:, 1] *= 0.05  # collapse second dimension

    # Run gradient descent on SIGReg loss
    _trajectory = [_emb.copy()]
    _losses = [sigreg_loss(_emb)]
    _ranks = [effective_rank(_emb)]

    _lr = 0.05
    for _step in range(_n_steps):
        # Numerical gradient (finite differences on each embedding coordinate)
        _grad = np.zeros_like(_emb)
        _eps = 0.01
        _base_loss = sigreg_loss(_emb, num_slices=64)
        for _j in range(_emb.shape[1]):
            _emb_p = _emb.copy()
            _emb_p[:, _j] += _eps
            _grad[:, _j] = (sigreg_loss(_emb_p, num_slices=64) - _base_loss) / _eps

        # Also add a "prediction" loss that pulls toward collapse
        _pred_grad = _emb * 0.3  # gradient of ||emb||^2 pulling toward origin

        # Combined gradient
        _total_grad = (1 - _lam) * _pred_grad + _lam * _grad / (_n + 1)
        _emb = _emb - _lr * _total_grad

        _trajectory.append(_emb.copy())
        _losses.append(sigreg_loss(_emb))
        _ranks.append(effective_rank(_emb))

    # Plot: initial, middle, final embeddings + loss curve
    fig3, axes3 = plt.subplots(1, 4, figsize=(14, 3.2))

    _show_idx = [0, len(_trajectory) // 2, -1]
    _labels = ["Initial (collapsed)", f"Step {len(_trajectory)//2}", f"Step {len(_trajectory)-1}"]

    for _idx, (_si, _lab) in enumerate(zip(_show_idx, _labels)):
        _data = _trajectory[_si]
        axes3[_idx].scatter(_data[:, 0], _data[:, 1], alpha=0.3, s=6, c="#4a90d9")
        axes3[_idx].set_xlim(-4, 4)
        axes3[_idx].set_ylim(-4, 4)
        axes3[_idx].set_aspect("equal")
        axes3[_idx].set_title(_lab, fontsize=10)
        axes3[_idx].text(
            0.05, 0.95,
            f"SIGReg: {_losses[_si]:.1f}\nRank: {_ranks[_si]:.2f}",
            transform=axes3[_idx].transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes3[3].plot(_losses, color="#2ecc71", linewidth=1.5)
    axes3[3].set_xlabel("Step")
    axes3[3].set_ylabel("SIGReg loss")
    axes3[3].set_title("Loss during optimization")

    _ax_rank = axes3[3].twinx()
    _ax_rank.plot(_ranks, color="#e74c3c", linewidth=1.5, linestyle="--")
    _ax_rank.set_ylabel("Effective rank", color="#e74c3c")
    _ax_rank.tick_params(axis="y", labelcolor="#e74c3c")

    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Characteristic function matching — what SIGReg actually computes

        The plot below shows the empirical characteristic function (ECF) of a random
        1D projection of the embeddings, compared against the Gaussian target
        $\phi(t) = e^{-t^2/2}$. SIGReg minimizes the shaded area between these curves.
        """
    )
    return


@app.cell
def _(mo):
    cf_dist_dropdown = mo.ui.dropdown(
        options={
            "Isotropic Gaussian": "iso",
            "Collapsed (rank-1)": "collapsed",
            "Uniform": "uniform",
            "Heavy-tailed (Cauchy)": "cauchy",
        },
        value="Isotropic Gaussian",
        label="Distribution"
    )
    mo.md(f"**Choose distribution:** {cf_dist_dropdown}")
    return (cf_dist_dropdown,)


@app.cell
def _(cf_dist_dropdown, np, plt, sigreg_loss):
    _rng = np.random.RandomState(123)
    _n, _d = 500, 8

    _dist = cf_dist_dropdown.value
    if _dist == "iso":
        _emb = _rng.randn(_n, _d)
    elif _dist == "collapsed":
        _emb = _rng.randn(_n, _d)
        _emb[:, 1:] *= 0.01
    elif _dist == "uniform":
        _emb = _rng.uniform(-2, 2, (_n, _d))
    elif _dist == "cauchy":
        _emb = _rng.standard_cauchy((_n, _d))
        _emb = np.clip(_emb, -10, 10)  # clip for visualization

    # Project onto one random direction
    _w = _rng.randn(_d)
    _w = _w / np.linalg.norm(_w)
    _proj = _emb @ _w

    # Compute ECF
    _t = np.linspace(-4, 4, 200)
    _ecf_real = np.mean(np.cos(_proj[:, None] * _t[None, :]), axis=0)
    _ecf_imag = np.mean(np.sin(_proj[:, None] * _t[None, :]), axis=0)
    _phi_gauss = np.exp(-_t**2 / 2.0)

    fig_cf, axes_cf = plt.subplots(1, 2, figsize=(11, 3.5))

    # Real part
    axes_cf[0].plot(_t, _phi_gauss, "k--", linewidth=1.5, label="Gaussian CF")
    axes_cf[0].plot(_t, _ecf_real, color="#4a90d9", linewidth=1.2, label="Empirical CF (real)")
    axes_cf[0].fill_between(
        _t, _phi_gauss, _ecf_real, alpha=0.2, color="#e74c3c",
        label="SIGReg penalty"
    )
    axes_cf[0].set_xlabel("t")
    axes_cf[0].set_ylabel("Re[CF(t)]")
    axes_cf[0].set_title("Real part of characteristic function")
    axes_cf[0].legend(fontsize=8)

    # Imaginary part (should be ~0 for symmetric distributions)
    axes_cf[1].axhline(0, color="k", linestyle="--", linewidth=1.5, label="Gaussian CF (= 0)")
    axes_cf[1].plot(_t, _ecf_imag, color="#4a90d9", linewidth=1.2, label="Empirical CF (imag)")
    axes_cf[1].fill_between(
        _t, 0, _ecf_imag, alpha=0.2, color="#e74c3c",
        label="SIGReg penalty"
    )
    axes_cf[1].set_xlabel("t")
    axes_cf[1].set_ylabel("Im[CF(t)]")
    axes_cf[1].set_title("Imaginary part of characteristic function")
    axes_cf[1].legend(fontsize=8)

    _loss = sigreg_loss(_emb)
    fig_cf.suptitle(
        f"SIGReg loss = {_loss:.2f}", fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig_cf
    return


# ============================================================
# SECTION 4: Old Heuristics vs SIGReg
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Old Heuristics vs. SIGReg — Side by Side

        Before LeJEPA, every JEPA variant needed its own bag of tricks to prevent
        collapse. Here is what each heuristic does and why SIGReg makes it redundant:

        | Heuristic | What it prevents | How it works | Why SIGReg replaces it |
        |-----------|-----------------|--------------|----------------------|
        | **Stop-gradient** | Target encoder updating toward trivial solution | Blocks gradients to one branch | SIGReg penalizes the trivial solution directly |
        | **Teacher-student (EMA)** | Rapid collapse of the target | Slowly tracks the student encoder | SIGReg enforces a fixed target distribution |
        | **Centering + sharpening** | Mean collapse + uniform collapse | Subtracts batch mean, sharpens logits | SIGReg enforces zero mean and unit variance by construction |
        | **Variance-covariance reg** | Dimensional collapse | Penalizes low variance and high covariance | SIGReg enforces isotropic covariance (superset) |
        | **Predictor network** | Direct copying between branches | Adds capacity to absorb non-trivial mapping | Not needed when the distribution target is enforced |

        The key insight: **all these heuristics are partial fixes for the same underlying
        problem.** They each prevent one failure mode. SIGReg prevents all failure modes
        simultaneously by enforcing the theoretically optimal distribution.

        Below, we train a tiny encoder (2-layer MLP, 16-dim embeddings) on synthetic
        image-like data, with and without SIGReg. Watch the embeddings.
        """
    )
    return


@app.cell
def _(effective_rank, np, plt, sigreg_loss):
    def _run_toy_training(use_sigreg, seed=0):
        """Train a tiny MLP encoder on synthetic data, return embedding trajectory."""
        rng = np.random.RandomState(seed)

        # Synthetic "images": 10 clusters in 32-dim input space
        n_per_class = 50
        n_classes = 8
        centers = rng.randn(n_classes, 32) * 3
        X = np.vstack([
            centers[i] + rng.randn(n_per_class, 32) * 0.5
            for i in range(n_classes)
        ])
        labels = np.repeat(np.arange(n_classes), n_per_class)

        # Tiny MLP: 32 -> 64 -> 16
        d_in, d_hid, d_out = 32, 64, 16
        W1 = rng.randn(d_in, d_hid) * 0.1
        b1 = np.zeros(d_hid)
        W2 = rng.randn(d_hid, d_out) * 0.1
        b2 = np.zeros(d_out)

        def forward(x):
            h = np.maximum(0, x @ W1 + b1)  # ReLU
            return h @ W2 + b2

        snapshots = []
        sigreg_losses = []
        ranks = []
        lr = 0.001

        for epoch in range(80):
            emb = forward(X)
            snapshots.append(emb.copy())
            sigreg_losses.append(sigreg_loss(emb, num_slices=32))
            ranks.append(effective_rank(emb))

            # Prediction loss: pull augmented views together (simulate with noise)
            X_aug = X + rng.randn(*X.shape) * 0.3
            emb_aug = forward(X_aug)
            pred_loss_grad_emb = 2 * (emb - emb_aug) / len(X)

            # Backprop through MLP (manual, for the toy demo)
            h = np.maximum(0, X @ W1 + b1)
            dL_demb = pred_loss_grad_emb

            if use_sigreg:
                # Approximate SIGReg gradient via finite differences on W2
                eps = 0.005
                base_sig = sigreg_loss(emb, num_slices=32)
                dW2_sig = np.zeros_like(W2)
                for j in range(min(d_out, 4)):  # Only first 4 dims for speed
                    W2_p = W2.copy()
                    W2_p[:, j] += eps
                    emb_p = np.maximum(0, X @ W1 + b1) @ W2_p + b2
                    dW2_sig[:, j] = (sigreg_loss(emb_p, num_slices=32) - base_sig) / eps

            # Gradient for W2, b2
            dW2 = h.T @ dL_demb / len(X)
            db2 = np.mean(dL_demb, axis=0)
            if use_sigreg:
                dW2 = dW2 + 0.3 * dW2_sig / (len(X) + 1)

            # Gradient for W1, b1
            dh = dL_demb @ W2.T
            dh = dh * (h > 0)  # ReLU grad
            dW1 = X.T @ dh / len(X)
            db1 = np.mean(dh, axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        return snapshots, sigreg_losses, ranks, labels

    _snap_no, _loss_no, _rank_no, _labels = _run_toy_training(False, seed=42)
    _snap_yes, _loss_yes, _rank_yes, _ = _run_toy_training(True, seed=42)

    # Visualization
    fig4, axes4 = plt.subplots(2, 4, figsize=(14, 6))

    _show_epochs = [0, 20, 50, 79]
    _cmap = plt.cm.Set2

    for col, ep in enumerate(_show_epochs):
        for row, (_snaps, _title_prefix) in enumerate([
            (_snap_no, "Without SIGReg"),
            (_snap_yes, "With SIGReg"),
        ]):
            ax = axes4[row, col]
            # PCA to 2D for visualization
            _data = _snaps[ep]
            _data_centered = _data - _data.mean(axis=0)
            try:
                U, S, Vt = np.linalg.svd(_data_centered, full_matrices=False)
                _proj2d = _data_centered @ Vt[:2].T
            except Exception:
                _proj2d = _data_centered[:, :2]

            ax.scatter(
                _proj2d[:, 0], _proj2d[:, 1],
                c=_labels, cmap="Set2", alpha=0.5, s=8
            )
            _lim = max(np.abs(_proj2d).max() * 1.2, 0.1)
            ax.set_xlim(-_lim, _lim)
            ax.set_ylim(-_lim, _lim)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"Epoch {ep}", fontsize=10)
            if col == 0:
                ax.set_ylabel(_title_prefix, fontsize=10, fontweight="bold")

    fig4.suptitle(
        "Training a toy encoder: collapse without SIGReg vs. healthy embeddings with SIGReg",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig4
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The top row shows collapse: by epoch 50, all clusters have merged.
        The bottom row (with SIGReg) maintains cluster separation because the regularizer
        forces the embedding distribution to stay spread across all dimensions.

        This is LeJEPA's core practical contribution: **one loss term replaces five separate heuristics.**
        """
    )
    return


# ============================================================
# SECTION 5: Extension — SIGReg + Compression Progress
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Extension: SIGReg Meets Intrinsic Motivation in World Models

        _This section presents novel empirical findings from our research on intrinsic
        motivation in JEPA latent spaces (Ashraf, 2026)._

        LeJEPA's SIGReg enforces that the encoder's embeddings follow an isotropic
        Gaussian distribution. But what happens to that distribution when the encoder
        encounters inputs it has never seen?

        We investigated this using **LeWM** (Maes & Le Lidec, 2026), an 18M-parameter
        world model that applies LeJEPA's architecture to continuous control tasks.
        Specifically, we asked: when we inject pixel noise into observations, does the
        encoder's latent distribution stay isotropic, or does SIGReg's constraint break?

        The answer connects LeJEPA's theory to a practical problem in reinforcement
        learning: **how curiosity-driven agents identify what is worth exploring.**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Finding 1: The encoder collapses global noise, but not localized noise

        When we injected full-frame pixel noise into Crafter observations (a 2D
        survival environment), the JEPA encoder mapped noisy inputs to a compact,
        low-dimensional latent cluster. But when we injected noise into a *small patch*
        of the image, the encoder did not discard it.

        This means the classic "noisy TV problem" in curiosity-driven RL (agents get
        stuck watching a noisy TV because it always looks novel) is partially solved by
        JEPA encoders for global noise, but *persists* for localized distractors.
        """
    )
    return


@app.cell
def _(np, plt):
    # ---- Hardcoded experimental data from the sprint ----
    # Source: technical_note_v2.md, Table in Section 3.1

    _conditions = ["Clean", "Gaussian\nnoise", "Pure\nrandom"]

    _eff_ranks = [94.8, 39.2, 10.1]
    _rnd_novelty = [0.014, 0.011, 0.012]
    _pred_loss = [0.014, 0.568, 0.640]

    fig5a, axes5a = plt.subplots(1, 3, figsize=(13, 3.5))

    # Effective rank
    _bars0 = axes5a[0].bar(
        _conditions, _eff_ranks,
        color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.7
    )
    axes5a[0].set_ylabel("Effective rank (of 192)")
    axes5a[0].set_title("Latent space dimensionality")
    axes5a[0].set_ylim(0, 110)
    for _b, _v in zip(_bars0, _eff_ranks):
        axes5a[0].text(
            _b.get_x() + _b.get_width()/2, _v + 2, f"{_v:.1f}",
            ha="center", fontsize=10, fontweight="bold"
        )

    # RND novelty
    _bars1 = axes5a[1].bar(
        _conditions, _rnd_novelty,
        color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.7
    )
    axes5a[1].set_ylabel("RND novelty")
    axes5a[1].set_title("Prediction-error novelty")
    axes5a[1].set_ylim(0, 0.020)
    for _b, _v in zip(_bars1, _rnd_novelty):
        axes5a[1].text(
            _b.get_x() + _b.get_width()/2, _v + 0.0005, f"{_v:.3f}",
            ha="center", fontsize=10
        )

    # Prediction loss
    _bars2 = axes5a[2].bar(
        _conditions, _pred_loss,
        color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.7
    )
    axes5a[2].set_ylabel("Prediction loss")
    axes5a[2].set_title("World model prediction loss")
    for _b, _v in zip(_bars2, _pred_loss):
        axes5a[2].text(
            _b.get_x() + _b.get_width()/2, _v + 0.01, f"{_v:.3f}",
            ha="center", fontsize=10
        )

    fig5a.suptitle(
        "JEPA encoder collapses global noise: effective rank drops 9.4x, but RND doesn't notice",
        fontsize=11, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    fig5a
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The effective rank drops from 94.8 (clean) to 10.1 (pure random noise) — a 9.4x
        reduction. The encoder compresses global noise into ~10 dimensions out of 192.
        Crucially, **RND novelty does not flag noise as interesting** (0.012 vs 0.014 for
        clean data). The encoder has already solved the noisy TV problem before the
        novelty signal even sees the data.

        But prediction loss tells a different story: 0.640 for noise vs. 0.014 for clean.
        The predictor cannot predict noise embeddings because they are out-of-distribution,
        even though they are low-dimensional. This is the key: **SIGReg's isotropic Gaussian
        target is violated by OOD inputs.**

        ### Finding 2: Localized noise breaks the collapse — dose-response curve

        The global collapse does not extend to localized noise. Below is a dose-response
        curve showing what happens as we increase noise patch size from 6% to 100%
        of the image:
        """
    )
    return


@app.cell
def _(mo):
    metric_dropdown = mo.ui.dropdown(
        options={
            "RND novelty": "rnd",
            "Prediction loss": "pred",
            "Compression progress": "cp",
            "Latent variance": "var",
            "Cosine similarity to clean": "cos",
        },
        value="Prediction loss",
        label="Metric to display"
    )
    mo.md(f"**Select metric:** {metric_dropdown}")
    return (metric_dropdown,)


@app.cell
def _(metric_dropdown, np, plt):
    # Hardcoded dose-response data from technical_note_v2.md Section 3.2
    _coverages = [0, 6, 25, 56, 100]  # percent of image covered by noise
    _coverage_labels = ["Clean\n(0%)", "16x16\n(6%)", "32x32\n(25%)", "48x48\n(56%)", "Global\n(100%)"]

    _data_dict = {
        "rnd":  [0.010, 0.013, 0.016, 0.016, 0.016],
        "pred": [0.008, 0.057, 0.017, 0.029, 0.024],
        "cp":   [0.002, 0.025, 0.009, 0.016, 0.012],
        "var":  [0.743, 0.135, 0.036, 0.009, 0.005],
        "cos":  [1.0, 0.41, 0.25, -0.05, -0.16],
    }

    _ylabel_dict = {
        "rnd": "RND novelty",
        "pred": "Prediction loss",
        "cp": "Compression progress",
        "var": "Latent variance",
        "cos": "Cosine similarity to clean",
    }

    _metric = metric_dropdown.value
    _vals = _data_dict[_metric]

    fig5b, ax5b = plt.subplots(figsize=(8, 4))

    _colors5 = ["#2ecc71", "#e67e22", "#e74c3c", "#c0392b", "#8e44ad"]
    _bars = ax5b.bar(
        _coverage_labels, _vals,
        color=_colors5, alpha=0.75, edgecolor="white", linewidth=0.8
    )

    # Highlight the non-monotonic peak if prediction loss or CP
    if _metric in ("pred", "cp"):
        _peak_idx = np.argmax(_vals)
        _bars[_peak_idx].set_edgecolor("#000")
        _bars[_peak_idx].set_linewidth(2.5)
        ax5b.annotate(
            "Non-monotonic\npeak", xy=(_peak_idx, _vals[_peak_idx]),
            xytext=(_peak_idx + 1.5, _vals[_peak_idx] * 0.95),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, ha="center",
        )

    ax5b.set_ylabel(_ylabel_dict[_metric])
    ax5b.set_xlabel("Noise patch size (% of image)")
    ax5b.set_title(
        f"Localized noise dose-response: {_ylabel_dict[_metric]}",
        fontsize=11, fontweight="bold"
    )

    for _b, _v in zip(_bars, _vals):
        ax5b.text(
            _b.get_x() + _b.get_width()/2,
            _v + (max(_vals) - min(_vals)) * 0.03,
            f"{_v:.3f}", ha="center", fontsize=9
        )

    plt.tight_layout()
    fig5b
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The non-monotonic finding

        The smallest noise patch (16x16, 6% of the image, just 4 ViT tokens) produces
        the **highest** prediction loss and compression progress. This is counterintuitive:
        more noise does not mean more disruption.

        The mechanism: with global noise, the encoder collapses everything to a compact
        region the predictor can partially fit. With a small patch, the encoder produces
        a **hybrid clean/noisy representation** that is harder to predict than either extreme.
        The non-monotonic response reveals that SIGReg's isotropic Gaussian structure is
        most disrupted not by maximal noise, but by the boundary between clean and corrupted
        representations.

        ### What this means for LeJEPA

        SIGReg enforces a specific distributional structure on the latent space. Our findings
        show that this structure is informative beyond its original purpose (preventing collapse):

        1. **OOD detection for free.** Deviations from isotropic Gaussian structure (measurable
           via effective rank, latent variance, or SIGReg loss itself) flag out-of-distribution
           inputs without requiring any additional machinery.

        2. **The encoder acts as a noise filter.** SIGReg-trained encoders compress global noise
           to low-rank representations that RND-style novelty detectors correctly ignore.
           This is a "built-in" partial solution to the noisy TV problem.

        3. **Localized distractors remain a problem.** The filtering does not extend to small
           noise patches. Agents using JEPA world models in environments with localized visual
           distractors still need additional mechanisms.

        These findings connect LeJEPA's theoretical framework to a practical question in
        reinforcement learning: *which states are worth exploring?* The answer depends on
        the latent space structure that SIGReg creates.

        ---

        **References**

        - Balestriero, R. & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised
          Learning Without the Heuristics. arXiv:2511.08544.
        - Maes, L. & Le Lidec, Q. (2026). LeWorldModel: World Modeling with Latent Imagination.
          arXiv:2603.19312.
        - Ashraf, S. (2026). JEPA Encoders Collapse Pixel Noise: Implications for Intrinsic
          Motivation in Learned Latent Spaces. In preparation for ICML 2026 RLxF Workshop.
        - Schmidhuber, J. (2009). Driven by Compression Progress. Springer.
        - Burda, Y. et al. (2018). Exploration by Random Network Distillation. ICLR 2019.
        """
    )
    return


if __name__ == "__main__":
    app.run()
