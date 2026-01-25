from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import y0_zeros
import torch
import sympy as sp
from sympy.stats import Beta, Kumaraswamy, density

class PerformancePlotUtils:

    @staticmethod
    def plot_performance(df:pd.DataFrame):
        X_axis_trials = df.index
        function_cols = ['f1', 'f2','f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        fig, ax = plt.subplots(figsize=(10, 10))

        for col in function_cols:
            y_values = df[col]
            ax.plot(X_axis_trials, y_values, marker='o', linestyle='-', markersize=4, alpha=0.8, label=col)
            line_color = ax.lines[-1].get_color()
        
            # Iterate over all (x, y) coordinates for the current line
            for x, y in zip(X_axis_trials, y_values):
                # Add text annotation near the marker
                ax.text(
                    x, 
                    y, 
                    f'{y:.2f}', # Format the value to 2 decimal places
                    color=line_color, 
                    fontsize=7, # Use a small font to reduce clutter
                    ha='right', # Align text horizontally to the right of the marker
                    va='bottom', # Align text vertically below the marker
                    alpha=0.7 # Slight transparency for better visual separation
                )
        ax.legend(title='Function')
        plt.title('Performance over trials')
        plt.xlabel('trials')
        plt.ylabel('Performance')
        plt.show()

    @torch.no_grad()
    @staticmethod
    def plot_3d_with_candidate_botorch(
        df: pd.DataFrame,
        candidate_str: str,                # e.g. "0.611316-0.808224"
        *,
        model,                             # BoTorch model: SingleTaskGP / your WrapperHeteroskedasticModel
        bounds=((0.0, 1.0), (0.0, 1.0)),   # (x1_min,max), (x2_min,max) in MODEL'S input space
        grid_n: int = 60,
        title: str = "BO surface (z = posterior mean). Points=obs, star=candidate",
        cmap: str = "viridis",
        show_surface: bool = True,
        show_contours: bool = True,
        observation_noise: bool = False,   # False -> latent f; True -> includes noise variance
    ):
        """
        df must have columns ['x1','x2','y'] already in the model's input/output units.
        If your model has outcome_transform=Standardize, BoTorch posteriors are returned in ORIGINAL y-units.
        """
        # --- parse candidate string ---
        try:
            cx, cy = (float(tok) for tok in candidate_str.split("-"))
        except Exception as e:
            raise ValueError(f"candidate_str should be 'x1-x2', got {candidate_str!r}") from e
        cand_np = np.array([cx, cy], dtype=float).reshape(1, 2)

        # --- data ---
        X_np = df[['x1', 'x2']].to_numpy(dtype=float)  # (n,2)
        y_np = df['y'].to_numpy(dtype=float)           # (n,)

        # --- device / dtype from model ---
        try:
            device = next(model.parameters()).device
            dtype  = next(model.parameters()).dtype
        except StopIteration:
            # Some wrappers may not expose parameters; fall back
            device = torch.device("cpu")
            dtype  = torch.double

        # --- grid for surface ---
        MU = STD = XX = YY = None
        if show_surface:
            xs = np.linspace(bounds[0][0], bounds[0][1], grid_n)
            ys = np.linspace(bounds[1][0], bounds[1][1], grid_n)
            XX, YY = np.meshgrid(xs, ys)
            G = np.c_[XX.ravel(), YY.ravel()]                          # (grid_n^2, 2)
            G_t = torch.as_tensor(G, dtype=dtype, device=device)       # to torch

            post = model.posterior(G_t, observation_noise=observation_noise)
            mu  = post.mean.view(-1).detach().cpu().numpy()
            std = post.variance.clamp_min(0).sqrt().view(-1).detach().cpu().numpy()
            MU  = mu.reshape(XX.shape)
            STD = std.reshape(XX.shape)

        # candidate predicted y (posterior mean)
        cand_t = torch.as_tensor(cand_np, dtype=dtype, device=device)
        post_c = model.posterior(cand_t, observation_noise=observation_noise)
        cand_mu = float(post_c.mean.view(-1).detach().cpu().numpy())

        # --- 3D plot ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        ax.set_title(title)

        # surface
        if MU is not None:
            surf = ax.plot_surface(XX, YY, MU, rstride=1, cstride=1,
                                cmap=cmap, alpha=0.6, linewidth=0, antialiased=True)
            cb = fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08)
            cb.set_label("Posterior mean")

        # observations
        sc = ax.scatter(X_np[:, 0], X_np[:, 1], y_np, s=40, c=y_np, cmap=cmap,
                        edgecolors="k", linewidths=0.4, label="observations")

        # candidate star (at posterior mean)
        ax.scatter(cand_np[0, 0], cand_np[0, 1], cand_mu, marker="*", s=400,
                edgecolor="k", linewidths=1.0, c="yellow",
                label=f"candidate μ={cand_mu:.3f}")
        # vertical line to surface for visual aid
        zmin = min(ax.get_zlim()[0], np.nanmin(MU) if MU is not None else cand_mu)
        ax.plot([cand_np[0, 0], cand_np[0, 0]], [cand_np[0, 1], cand_np[0, 1]],
                [zmin, cand_mu], "k--", alpha=0.5)

        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

        # Optional 2D contour plot of mean or std
        if show_contours and MU is not None:
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            c0 = ax2[0].contourf(XX, YY, MU, levels=20, cmap=cmap)
            fig2.colorbar(c0, ax=ax2[0]); ax2[0].set_title("Posterior mean")
            c1 = ax2[1].contourf(XX, YY, STD, levels=20, cmap=cmap)
            fig2.colorbar(c1, ax=ax2[1]); ax2[1].set_title("Posterior std")
            for a in ax2:
                a.scatter(X_np[:, 0], X_np[:, 1], s=20, c='k', alpha=0.6)
                a.scatter(cand_np[0, 0], cand_np[0, 1], marker="*", s=200, c="yellow", edgecolor="k")
                a.set_xlabel("x1"); a.set_ylabel("x2")
            plt.tight_layout(); plt.show()


    @torch.no_grad()
    def plot_pairwise_slices_botorch_3d(
        df,                                   # must contain ['x1','x2','x3','y'] in model units
        candidate_str,                         # e.g. "0.61-0.81-0.27"
        *,
        model,                                 # BoTorch model (SingleTaskGP or your wrapper)
        bounds=((0,1),(0,1),(0,1)),            # ((x1min,x1max),(x2min,x2max),(x3min,x3max))
        grid_n=60,
        show_surface=True,                     # 3D surface if True, else 2D contour
        show_std=False,                        # draw a second row with posterior std contours
        observation_noise=False,               # True -> include noise in posterior
        fixed_values=None,                     # dict like {'x1':0.3} if you want to override slice levels
        cmap="viridis",
        title_prefix="BO surface (z = posterior mean)",
    ):
    # ---- parse candidate
        try:
            c1, c2, c3 = (float(tok) for tok in candidate_str.split("-"))
        except Exception as e:
            raise ValueError(f"candidate_str should be 'x1-x2-x3', got {candidate_str!r}") from e
        cand = np.array([c1, c2, c3], dtype=float).reshape(1, 3)

        # ---- data
        X_np = df[['x1','x2','x3']].to_numpy(dtype=float)
        y_np = df['y'].to_numpy(dtype=float)

        # ---- device / dtype from model
        try:
            device = next(model.parameters()).device
            dtype  = next(model.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype  = torch.double

        # ---- helper to eval posterior mean/std on a 2D grid with one dim fixed
        def eval_on_grid(dim_pair, fixed_dim, fixed_val):
            # dim_pair: tuple of two dims (e.g., (0,1) for x1,x2)
            # fixed_dim: the remaining dim (e.g., 2 for x3)
            # fixed_val: scalar value to fix that dim at
            axes = [0,1,2]
            d1, d2 = dim_pair
            assert fixed_dim in axes and fixed_dim not in dim_pair

            xs = np.linspace(bounds[d1][0], bounds[d1][1], grid_n)
            ys = np.linspace(bounds[d2][0], bounds[d2][1], grid_n)
            XX, YY = np.meshgrid(xs, ys)

            G = np.zeros((grid_n*grid_n, 3), dtype=float)
            G[:, d1] = XX.ravel()
            G[:, d2] = YY.ravel()
            G[:, fixed_dim] = fixed_val

            G_t = torch.as_tensor(G, dtype=dtype, device=device)
            post = model.posterior(G_t, observation_noise=observation_noise)
            mu  = post.mean.view(-1).detach().cpu().numpy()
            std = post.variance.clamp_min(0).sqrt().view(-1).detach().cpu().numpy()
            MU  = mu.reshape(XX.shape)
            STD = std.reshape(XX.shape)
            return XX, YY, MU, STD

        # determine slice levels (fixed values) for each panel
        # default: use the candidate’s coordinate for the fixed dimension
        fixed_vals = {
            'x1': cand[0,0], 'x2': cand[0,1], 'x3': cand[0,2]
        }
        if fixed_values:
            fixed_vals.update(fixed_values)

        # panels: (x1,x2) @ x3 fixed; (x1,x3) @ x2 fixed; (x2,x3) @ x1 fixed
        panels = [
            ((0,1), 2, fixed_vals['x3'], "x1", "x2", "x3"),
            ((0,2), 1, fixed_vals['x2'], "x1", "x3", "x2"),
            ((1,2), 0, fixed_vals['x1'], "x2", "x3", "x1"),
        ]

        nrows = 2 if show_std else 1
        fig = plt.figure(figsize=(5*len(panels), 5*nrows))
        axes = []

        for p_idx, (dim_pair, fixed_dim, fv, xl, yl, zlname) in enumerate(panels):
            XX, YY, MU, STD = eval_on_grid(dim_pair, fixed_dim, fv)

            # candidate projections for this panel
            cx, cy = cand[0, dim_pair[0]], cand[0, dim_pair[1]]
            # Observations projected to this panel (only for display—z comes from y)
            obs_x = X_np[:, dim_pair[0]]
            obs_y = X_np[:, dim_pair[1]]

            # ---- Row 1: posterior mean
            if show_surface:
                ax = fig.add_subplot(nrows, len(panels), 1 + p_idx, projection="3d")
                ax.plot_surface(XX, YY, MU, rstride=1, cstride=1, cmap=cmap, alpha=0.65, linewidth=0)
                # scatter obs with true y on z-axis
                ax.scatter(obs_x, obs_y, y_np, s=25, c=y_np, cmap=cmap, edgecolors="k", linewidths=0.3, alpha=0.9, label="obs")
                # candidate star at posterior mean
                # interpolate candidate's mu by querying posterior directly to avoid grid mismatch
                c_full = np.zeros((1,3), dtype=float); c_full[:] = cand
                c_t = torch.as_tensor(c_full, dtype=dtype, device=device)
                mu_c = float(model.posterior(c_t, observation_noise=observation_noise).mean.view(-1).cpu().numpy())
                ax.scatter(cx, cy, mu_c, marker="*", s=300, c="yellow", edgecolor="k", linewidths=0.8, label=f"cand μ={mu_c:.3f}")
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_zlabel("y")
                ax.set_title(f"{title_prefix}  |  fix {zlname}={fv:.3f}")
                ax.legend(loc="upper left")
                axes.append(ax)
            else:
                ax = fig.add_subplot(nrows, len(panels), 1 + p_idx)
                c0 = ax.contourf(XX, YY, MU, levels=20, cmap=cmap)
                fig.colorbar(c0, ax=ax)
                ax.scatter(obs_x, obs_y, s=20, c='k', alpha=0.6)
                ax.scatter(cx, cy, marker="*", s=150, c="yellow", edgecolor="k")
                ax.set_xlabel(xl); ax.set_ylabel(yl)
                ax.set_title(f"Posterior mean | fix {zlname}={fv:.3f}")
                axes.append(ax)

            # ---- Row 2: posterior std (optional)
            if show_std:
                if show_surface:
                    ax2 = fig.add_subplot(nrows, len(panels), len(panels) + 1 + p_idx, projection="3d")
                    ax2.plot_surface(XX, YY, STD, rstride=1, cstride=1, cmap=cmap, alpha=0.85, linewidth=0)
                    ax2.set_xlabel(xl); ax2.set_ylabel(yl); ax2.set_zlabel("std")
                    ax2.set_title(f"Posterior std | fix {zlname}={fv:.3f}")
                else:
                    ax2 = fig.add_subplot(nrows, len(panels), len(panels) + 1 + p_idx)
                    c1 = ax2.contourf(XX, YY, STD, levels=20, cmap=cmap)
                    fig.colorbar(c1, ax=ax2)
                    ax2.scatter(obs_x, obs_y, s=18, c='k', alpha=0.5)
                    ax2.scatter(cx, cy, marker="*", s=120, c="yellow", edgecolor="k")
                    ax2.set_xlabel(xl); ax2.set_ylabel(yl)
                    ax2.set_title(f"Posterior std | fix {zlname}={fv:.3f}")
                axes.append(ax2)

        plt.tight_layout()
        plt.show()


    @torch.no_grad()
    def plot_pairwise_slices_botorch_4d(
        df,                                   # must contain ['x1','x2','x3','x4','y'] in model units
        candidate_str,                        # e.g. "0.2-0.5-0.7-0.1"
        *,
        model,                                # BoTorch model (SingleTaskGP or your wrapper)
        bounds=((0,1),(0,1),(0,1),(0,1)),     # ((x1min,x1max),...,(x4min,x4max))
        grid_n=60,
        show_surface=True,                    # 3D surface if True, else 2D contour
        show_std=False,                       # draw a second row with posterior std
        observation_noise=False,              # True -> include noise in posterior
        fixed_values=None,                    # dict like {'x1':0.3,'x4':0.8} to override slice levels
        cmap="viridis",
        title_prefix="BO surface (z = posterior mean)",
    ):
        # ---- parse candidate
        try:
            c1, c2, c3, c4 = (float(tok) for tok in candidate_str.split("-"))
        except Exception as e:
            raise ValueError(
                f"candidate_str should be 'x1-x2-x3-x4', got {candidate_str!r}"
            ) from e

        cand = np.array([c1, c2, c3, c4], dtype=float).reshape(1, 4)

        # ---- data
        X_np = df[['x1', 'x2', 'x3', 'x4']].to_numpy(dtype=float)
        y_np = df['y'].to_numpy(dtype=float)

        # ---- device / dtype from model
        try:
            p = next(model.parameters())
            device = p.device
            dtype = p.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.double

        # ---- default fixed values: candidate coordinates
        #     stored by dimension index 0..3
        dim_names = ["x1", "x2", "x3", "x4"]
        fixed_vals_by_dim = {i: cand[0, i] for i in range(4)}

        # override from fixed_values (specified in terms of 'x1', 'x2', etc.)
        if fixed_values:
            name_to_idx = {name: i for i, name in enumerate(dim_names)}
            for k, v in fixed_values.items():
                if k not in name_to_idx:
                    raise KeyError(f"Unknown dimension name {k!r}, use one of {dim_names}")
                fixed_vals_by_dim[name_to_idx[k]] = float(v)

        # ---- helper: eval posterior mean/std on a 2D grid, other dims fixed
        def eval_on_grid(dim_pair, fixed_vals_by_dim):
            d1, d2 = dim_pair
            xs = np.linspace(bounds[d1][0], bounds[d1][1], grid_n)
            ys = np.linspace(bounds[d2][0], bounds[d2][1], grid_n)
            XX, YY = np.meshgrid(xs, ys)

            G = np.zeros((grid_n * grid_n, 4), dtype=float)
            G[:, d1] = XX.ravel()
            G[:, d2] = YY.ravel()

            # set the other two dimensions to their fixed values
            for fd in range(4):
                if fd not in dim_pair:
                    G[:, fd] = fixed_vals_by_dim[fd]

            G_t = torch.as_tensor(G, dtype=dtype, device=device)
            post = model.posterior(G_t, observation_noise=observation_noise)

            mu = post.mean.view(-1).detach().cpu().numpy()
            var = post.variance.clamp_min(0).view(-1).detach().cpu().numpy()
            std = np.sqrt(var)

            MU = mu.reshape(XX.shape)
            STD = std.reshape(XX.shape)
            return XX, YY, MU, STD

        # candidate posterior mean (same for all panels)
        c_t = torch.as_tensor(cand, dtype=dtype, device=device)
        mu_c = float(model.posterior(c_t, observation_noise=observation_noise).mean
                    .view(-1).detach().cpu().numpy())

        # ---- all 2D pairs out of 4 dims
        dim_pairs = list(combinations(range(4), 2))   # [(0,1), (0,2), ..., (2,3)]
        n_panels = len(dim_pairs)

        nrows = 2 if show_std else 1
        fig = plt.figure(figsize=(5 * n_panels, 5 * nrows))
        axes = []

        for p_idx, dim_pair in enumerate(dim_pairs):
            d1, d2 = dim_pair
            fixed_dims = [d for d in range(4) if d not in dim_pair]

            XX, YY, MU, STD = eval_on_grid(dim_pair, fixed_vals_by_dim)

            # projection of candidate & data onto current 2D plane
            cx, cy = cand[0, d1], cand[0, d2]
            obs_x = X_np[:, d1]
            obs_y = X_np[:, d2]

            xl = dim_names[d1]
            yl = dim_names[d2]

            # for title: show what is fixed
            fixed_str = ", ".join(
                f"{dim_names[d]}={fixed_vals_by_dim[d]:.3f}" for d in fixed_dims
            )

            # ---- Row 1: posterior mean
            if show_surface:
                ax = fig.add_subplot(nrows, n_panels, 1 + p_idx, projection="3d")
                ax.plot_surface(XX, YY, MU, rstride=1, cstride=1,
                                cmap=cmap, alpha=0.65, linewidth=0)

                # scatter observations (z = true y)
                ax.scatter(
                    obs_x, obs_y, y_np,
                    s=25, c=y_np, cmap=cmap,
                    edgecolors="k", linewidths=0.3, alpha=0.9, label="obs"
                )

                # candidate star at posterior mean
                ax.scatter(
                    cx, cy, mu_c,
                    marker="*", s=300, c="yellow",
                    edgecolor="k", linewidths=0.8,
                    label=f"cand μ={mu_c:.3f}"
                )

                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_zlabel("y")
                ax.set_title(f"{title_prefix} | fix {fixed_str}")
                ax.legend(loc="upper left")
                axes.append(ax)
            else:
                ax = fig.add_subplot(nrows, n_panels, 1 + p_idx)
                c0 = ax.contourf(XX, YY, MU, levels=20, cmap=cmap)
                fig.colorbar(c0, ax=ax)
                ax.scatter(obs_x, obs_y, s=20, c='k', alpha=0.6, label="obs")
                ax.scatter(cx, cy, marker="*", s=150, c="yellow", edgecolor="k",
                        label=f"cand μ={mu_c:.3f}")
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_title(f"Posterior mean | fix {fixed_str}")
                ax.legend(loc="upper left")
                axes.append(ax)

            # ---- Row 2: posterior std
            if show_std:
                if show_surface:
                    ax2 = fig.add_subplot(nrows, n_panels, n_panels + 1 + p_idx,
                                        projection="3d")
                    ax2.plot_surface(XX, YY, STD, rstride=1, cstride=1,
                                    cmap=cmap, alpha=0.85, linewidth=0)
                    ax2.set_xlabel(xl)
                    ax2.set_ylabel(yl)
                    ax2.set_zlabel("std")
                    ax2.set_title(f"Posterior std | fix {fixed_str}")
                else:
                    ax2 = fig.add_subplot(nrows, n_panels, n_panels + 1 + p_idx)
                    c1 = ax2.contourf(XX, YY, STD, levels=20, cmap=cmap)
                    fig.colorbar(c1, ax=ax2)
                    ax2.scatter(obs_x, obs_y, s=18, c='k', alpha=0.5)
                    ax2.scatter(cx, cy, marker="*", s=120, c="yellow", edgecolor="k")
                    ax2.set_xlabel(xl)
                    ax2.set_ylabel(yl)
                    ax2.set_title(f"Posterior std | fix {fixed_str}")
                axes.append(ax2)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_3slice_surfaces_with_candidate(
        df: pd.DataFrame,
        candidate_str: str,                 # "x1-x2-x3"
        *,
        gp=None,                            # optional fitted sklearn GPR
        bounds=((0.0,1.0),(0.0,1.0),(0.0,1.0)),
        grid_n: int = 60,
        title: str = "GP posterior mean slices (z = y)",
        cmap: str = "viridis",
    ):
        """
        Plots three 3D surfaces (z=y) as slices of y=f(x1,x2,x3):
        - z over (x1,x2) at x3=c3
        - z over (x1,x3) at x2=c2
        - z over (x2,x3) at x1=c1
        Highlights the candidate (star) in each slice.
        df must have columns: ['x1','x2','x3','y'] with x in [0,1].
        """
        # --- parse candidate ---
        try:
            c1, c2, c3 = (float(tok) for tok in candidate_str.split("-"))
        except Exception as e:
            raise ValueError(f"candidate_str should be 'x1-x2-x3', got {candidate_str!r}") from e

        X = df[['x1','x2','x3']].to_numpy(float)
        Y = df['y'].to_numpy(float)

        # helper: grid -> predict mean
        def surface_over(axes=("x1","x2"), fixed=("x3", c3)):
            # build mesh
            a, b = axes
            xs = np.linspace(bounds[0][0], bounds[0][1], grid_n)
            ys = np.linspace(bounds[1][0], bounds[1][1], grid_n)
            zs = np.linspace(bounds[2][0], bounds[2][1], grid_n)
            name2idx = {"x1":0,"x2":1,"x3":2}
            A, B = np.meshgrid(
                xs if a=="x1" else (ys if a=="x2" else zs),
                xs if b=="x1" else (ys if b=="x2" else zs),
            )
            G = np.zeros((A.size, 3), dtype=float)
            # fill columns
            for name in ("x1","x2","x3"):
                if name == a:
                    G[:, name2idx[name]] = A.ravel()
                elif name == b:
                    G[:, name2idx[name]] = B.ravel()
                else:
                    # fixed
                    fname, fval = fixed
                    assert name == fname
                    G[:, name2idx[name]] = fval
            MU = None
            if gp is not None:
                MU = gp.predict(G, return_std=False).reshape(A.shape)
            return A, B, MU

        # prepare three slices
        A12, B12, MU12 = surface_over(("x1","x2"), ("x3", c3))
        A13, B13, MU13 = surface_over(("x1","x3"), ("x2", c2))
        A23, B23, MU23 = surface_over(("x2","x3"), ("x1", c1))

        fig = plt.figure(figsize=(16, 4.8))
        fig.suptitle(title, fontsize=13)

        # --- (x1,x2) @ x3=c3 ---
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("y")
        ax1.set_title(f"(x1,x2) slice @ x3={c3:.3f}")
        if MU12 is not None:
            surf = ax1.plot_surface(A12, B12, MU12, cmap=cmap, alpha=0.7, linewidth=0, antialiased=True)
        # scatter observed points projected to this slice (drop true x3, use their own y)
        ax1.scatter(X[:,0], X[:,1], Y, c=Y, cmap=cmap, s=24, edgecolors="k", linewidths=0.3, alpha=0.9)
        # candidate star
        if gp is not None:
            z_star = float(gp.predict(np.array([[c1,c2,c3]]), return_std=False))
        else:
            # nearest observed
            d2 = np.sum((X - np.array([c1,c2,c3]))**2, axis=1)
            z_star = float(Y[np.argmin(d2)])
        ax1.scatter([c1],[c2],[z_star], marker="*", s=240, c="yellow", edgecolor="k", linewidths=0.8)

        # --- (x1,x3) @ x2=c2 ---
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.set_xlabel("x1"); ax2.set_ylabel("x3"); ax2.set_zlabel("y")
        ax2.set_title(f"(x1,x3) slice @ x2={c2:.3f}")
        if MU13 is not None:
            ax2.plot_surface(A13, B13, MU13, cmap=cmap, alpha=0.7, linewidth=0, antialiased=True)
        ax2.scatter(X[:,0], X[:,2], Y, c=Y, cmap=cmap, s=24, edgecolors="k", linewidths=0.3, alpha=0.9)
        ax2.scatter([c1],[c3],[z_star], marker="*", s=240, c="yellow", edgecolor="k", linewidths=0.8)

        # --- (x2,x3) @ x1=c1 ---
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.set_xlabel("x2"); ax3.set_ylabel("x3"); ax3.set_zlabel("y")
        ax3.set_title(f"(x2,x3) slice @ x1={c1:.3f}")
        if MU23 is not None:
            ax3.plot_surface(A23, B23, MU23, cmap=cmap, alpha=0.7, linewidth=0, antialiased=True)
        ax3.scatter(X[:,1], X[:,2], Y, c=Y, cmap=cmap, s=24, edgecolors="k", linewidths=0.3, alpha=0.9)
        ax3.scatter([c2],[c3],[z_star], marker="*", s=240, c="yellow", edgecolor="k", linewidths=0.8)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_beta_kumaraswamy_distributions():

        def get_pdfs(a, b):
            # Define Beta and Kumaraswamy random variables with shape params a, b
            x = sp.symbols('x', real=True)
            B = Beta('B', a, b)
            K = Kumaraswamy('K', a, b)

            beta_pdf_expr = density(B)(x)          # symbolic PDF of Beta
            kuma_pdf_expr = density(K)(x)          # symbolic PDF of Kumaraswamy

            beta_pdf = sp.lambdify(x, beta_pdf_expr, 'numpy')
            kuma_pdf = sp.lambdify(x, kuma_pdf_expr, 'numpy')
            return beta_pdf, kuma_pdf
            
        params = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
        xs = np.linspace(0, 1, 400)
        plt.figure(figsize=(9, 6))
        for a, b in params:
            beta_pdf, kuma_pdf = get_pdfs(a, b)
            plt.plot(xs, beta_pdf(xs), label=f"Beta({a},{b})", linewidth=2)
            plt.plot(xs, kuma_pdf(xs), '--', label=f"Kumaraswamy({a},{b})", linewidth=1.4)

        plt.title("Beta (solid) vs Kumaraswamy (dashed) – SymPy-based PDFs")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_outputs_vs_distributions(y: np.ndarray):
        xs = np.linspace(0, 1, 200)
        a, b, x = sp.symbols('a b x', positive=True, real=True)
        B = Beta('B', a, b)
        K = Kumaraswamy('K', a, b)

        def beta_pdf_np(y_vals, a, b):
            from math import gamma
            y_vals = np.clip(y_vals, 1e-12, 1 - 1e-12)
            norm = gamma(a) * gamma(b) / gamma(a + b)
            return (y_vals ** (a - 1)) * ((1 - y_vals) ** (b - 1)) / norm

        def kuma_pdf_np(y_vals, a, b):
            y_vals = np.clip(y_vals, 1e-12, 1 - 1e-12)
            return a * b * (y_vals ** (a - 1)) * ((1 - y_vals ** a) ** (b - 1))

        def gaussian_nll(y_vals):
            y_vals = np.asarray(y_vals, dtype=float)
            mu_hat = np.mean(y_vals)
            sigma_hat = np.std(y_vals, ddof=0)

            # Gaussian log-pdf
            var = sigma_hat**2
            logpdf = -0.5 * np.log(2*np.pi*var) - 0.5 * ((y_vals - mu_hat)**2 / var)
            # Negative log-likelihood
            return -np.sum(logpdf), mu_hat, sigma_hat

        def neg_log_likelihood_beta(params):
            a_, b_ = params
            if a_ <= 0 or b_ <= 0:
                return np.inf
            pdf_vals = beta_pdf_np(y, a_, b_)
            pdf_vals = np.clip(pdf_vals, 1e-15, None)
            return -np.sum(np.log(pdf_vals))

        def neg_log_likelihood_kuma(params):
            a_, b_ = params
            if a_ <= 0 or b_ <= 0:
                return np.inf
            pdf_vals = kuma_pdf_np(y, a_, b_)
            pdf_vals = np.clip(pdf_vals, 1e-15, None)
            return -np.sum(np.log(pdf_vals))

    

        from scipy.optimize import minimize
        
        res_beta = minimize(neg_log_likelihood_beta, x0=[2, 2], bounds=[(1e-3, 10), (1e-3, 10)])
        res_kuma = minimize(neg_log_likelihood_kuma, x0=[2, 2], bounds=[(1e-3, 10), (1e-3, 10)])
        nll_beta = res_beta.fun
        nll_kuma= res_kuma.fun
        nll_gauss, mu_hat, sigma_hat = gaussian_nll(y)

        print(f"Log-likelihood Beta: { nll_beta:.2f}")
        print(f"Log-likelihood Kumaraswamy: { nll_kuma:.2f}")
        print(f"Gaussian NLL    : {nll_gauss:.4f} (μ={mu_hat:.3f}, σ={sigma_hat:.3f})")

        plt.hist(y, bins=10, density=True, alpha=0.7, label="data", color="steelblue", edgecolor="black")
        # plt.plot(xs, beta_pdf_np(xs, *res_beta.x), label="Beta fit", lw=2)
        # plt.plot(xs, beta_pdf_np(xs, *res_kuma.x), '--', label="Kumaraswamy fit", lw=2)
        plt.legend()
        plt.xlabel("scaled y")
        plt.ylabel("density")
        plt.title("Beta vs Kumaraswamy fits (SymPy-based PDFs)")
        plt.show()