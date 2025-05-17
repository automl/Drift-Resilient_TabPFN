import torch
import numpy as np

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba

from sklearn.utils import _safe_indexing
from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted


def validate_parameters(estimator, grid_resolution, eps, plot_method):
    check_is_fitted(estimator)
    if is_regressor(estimator):
        raise ValueError("Regressors are not supported")
    if grid_resolution <= 1:
        raise ValueError("grid_resolution must be greater than 1.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if plot_method not in {"contourf", "contour", "pcolormesh"}:
        raise ValueError(
            f"Invalid plot_method: {plot_method}. Choose from 'contourf', 'contour', 'pcolormesh'."
        )


def create_meshgrid(X, all_X, eps, grid_resolution):
    # If not all data points are provided separately, we scale the plot to the data points that will be scattered
    # in this call.
    if all_X is None:
        all_X = X

    # Get the min and max (with eps as additional spacing) for the scale of the plot.
    x0, x1 = _safe_indexing(all_X, 0, axis=1), _safe_indexing(all_X, 1, axis=1)

    x0_min, x0_max = x0.min(), x0.max()
    x1_min, x1_max = x1.min(), x1.max()

    min_glob = min(x0_min, x1_min) - eps
    max_glob = max(x0_max, x1_max) + eps

    # Now switch to the real data points, to scatter in this call.
    x0, x1 = _safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1)

    # Create a meshgrid for the whole plot surface.
    xx0, xx1 = np.meshgrid(
        np.linspace(min_glob, max_glob, grid_resolution),
        np.linspace(min_glob, max_glob, grid_resolution),
    )

    X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    return X_grid, xx0, xx1, x0, x1


def get_predictions(estimator, X_grid, dist_shift_domain):
    pred_func = getattr(estimator, "predict_proba")

    # For all samples in this plot, we set the dist_shift_domain to be the provided one.
    if dist_shift_domain is not None:
        dist_shift_domain = torch.tensor(
            np.full((X_grid.shape[0]), dist_shift_domain, dtype=np.float32)
        )

        preds = pred_func(X_grid, additional_x={"dist_shift_domain": dist_shift_domain})
    else:
        preds = pred_func(X_grid)

    return preds


def get_plot_visuals(num_classes):
    if num_classes <= 3:
        shading_colors = {0: "#AAAAAA", 1: "#5C87B4", 2: "#5E9048"}
        node_colors = {0: "#888888", 1: "#93BCEC", 2: "#9CA951"}
    else:
        colormap = colormaps["tab10"]
        shading_colors = {i: to_hex(colormap.colors[i]) for i in range(num_classes)}
        node_colors = {i: to_hex(colormap.colors[i + 1]) for i in range(num_classes)}

    markers = {i: (i + 3, i % 2, 0) for i in range(num_classes)}

    return shading_colors, node_colors, markers


def make_lighter_color(color_hex, alpha=0.2):
    # Convert hex to RGBA
    original_color = to_rgba(color_hex)
    # Blend with white, where alpha controls the intensity of the original color
    lighter_color = [1 - alpha * (1 - c) for c in original_color[:3]] + [
        original_color[3]
    ]
    return to_hex(lighter_color)


def plot_decision_boundary(
    estimator,
    X,
    *,
    grid_resolution=50,
    eps=1.0,
    plot_method="contourf",
    xlabel=None,
    ylabel=None,
    ax=None,
    y_gt=None,
    y_pred=None,
    dist_shift_domain=None,
    all_X=None,
    show_colorbar=True,
    show_legend=True,
    cbar_ax=None,
    legend_ax=None,
    **kwargs,
):
    check_is_fitted(estimator)

    validate_parameters(estimator, grid_resolution, eps, plot_method)

    X_grid, xx0, xx1, x0, x1 = create_meshgrid(X, all_X, eps, grid_resolution)

    preds = get_predictions(estimator, X_grid, dist_shift_domain)

    num_classes = preds.shape[1]
    kwargs["vmin"] = 0
    kwargs["vmax"] = 1

    shading_colors, node_colors, markers = get_plot_visuals(num_classes)

    max_probs = preds.max(axis=1).reshape(xx0.shape)
    winning_classes = preds.argmax(axis=1).reshape(xx0.shape)

    if plot_method in ("contour", "contourf"):
        kwargs["levels"] = np.linspace(1e-10, 1, 11)

    if ax is None:
        _, ax = plt.subplots()

    plot_func = getattr(ax, plot_method)

    for i in range(num_classes):
        prob_class = max_probs.copy()
        prob_class[winning_classes != i] = 0

        color_list = [
            (0, "#ffffffff"),
            (1e-10, make_lighter_color(shading_colors[i], 0.15)),
            (1, shading_colors[i]),
        ]
        cmap = LinearSegmentedColormap.from_list("", color_list, N=256)

        surface_ = plot_func(
            xx0, xx1, prob_class, cmap=cmap, alpha=0.85, antialiased=True, **kwargs
        )

        # Add the colorbar.
        if show_colorbar:
            if cbar_ax is None:
                ax.figure.colorbar(surface_)
            else:
                # Configure cbar_ax as a container by hiding its ticks, labels, and border
                cbar_ax.set_xticks([])
                cbar_ax.set_yticks([])
                cbar_ax.axis("off")

                cax = cbar_ax.inset_axes(
                    [i / num_classes, 0.0, 1 / num_classes, 1.0],
                    transform=cbar_ax.transAxes,
                )
                cbar = cbar_ax.figure.colorbar(surface_, cax=cax)

                if i < num_classes - 1:
                    # Remove labels for the first colorbars
                    cbar.ax.set_yticklabels([])
                else:
                    # Add label to the first colorbar
                    cbar.ax.set_ylabel("Prediction Confidence")

    ax.contour(
        xx0,
        xx1,
        winning_classes,
        colors="k",
        levels=[i for i in range(num_classes)],
        linewidths=1.5,
        linestyles="-",
        alpha=0.85,
    )

    y_pred = y_pred if y_pred is not None else y_gt

    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)

    if x0 is not None and x1 is not None and y_gt is not None:
        for i in range(num_classes):
            # Correctly classified points
            mask_correct = (y_gt == i) & (y_gt == y_pred)
            ax.scatter(
                x0[mask_correct],
                x1[mask_correct],
                s=70,
                marker=markers[i],
                color=node_colors[i],
                edgecolor="k",
                label=f"Class {i}",
                alpha=0.85,
            )

            # Misclassified points
            mask_incorrect = (y_gt == i) & (y_gt != y_pred)
            ax.scatter(
                x0[mask_incorrect],
                x1[mask_incorrect],
                s=70,
                marker=markers[i],
                color=(0.75, 0, 0),
                edgecolor="k",
                alpha=0.85,
            )

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=16)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16)

    # Add the legend.
    if show_legend:
        if legend_ax is None:
            ax.legend(fontsize=14)
        else:
            legend_ax.legend(
                *ax.get_legend_handles_labels(), loc="center left", fontsize=14
            )
            legend_ax.axis("off")
