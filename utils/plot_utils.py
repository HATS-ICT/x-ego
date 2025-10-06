import torch
import torch.nn.functional as F
import awpy
import awpy.data
import awpy.plot.utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.stats
import warnings
from matplotlib.figure import Figure, Axes
from .metric_utils import chamfer_distance_batch


def create_prediction_plots(task_form, predictions, targets, output_dir, pov_team_sides=None):
    """Create and save prediction analysis plots."""
    if task_form == 'regression':
        create_regression_plots(predictions, targets, output_dir, pov_team_sides)
    elif task_form == 'classification':
        create_classification_plots(predictions, targets, output_dir, pov_team_sides)

def create_regression_plots(predictions, targets, output_dir, pov_team_sides=None):
    """Create plots for regression predictions."""
    # predictions and targets shape: [N, 5, 2] (X,Y only)
    predictions_flat = predictions.reshape(-1, 10)  # [N, 10]
    targets_flat = targets.reshape(-1, 10)
    
    # Create scatter plots for each coordinate
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Enemy Location Prediction: Predicted vs Actual Coordinates (X,Y)', fontsize=16)
    
    for player in range(5):
        for coord_idx, coord_name in enumerate(['X', 'Y']):
            dim_idx = player * 2 + coord_idx
            
            ax = axes[coord_idx, player]
            
            # Plot with team-specific colors if available
            if pov_team_sides is not None:
                ct_mask = pov_team_sides == 'CT'
                t_mask = pov_team_sides == 'T'
                
                if np.any(ct_mask):
                    ax.scatter(targets_flat[ct_mask, dim_idx], predictions_flat[ct_mask, dim_idx], 
                                alpha=0.6, s=10, c='blue', label='CT', marker='o')
                if np.any(t_mask):
                    ax.scatter(targets_flat[t_mask, dim_idx], predictions_flat[t_mask, dim_idx], 
                                alpha=0.6, s=10, c='red', label='T', marker='^')
                
                if player == 0 and coord_idx == 0:  # Add legend only once
                    ax.legend()
            else:
                ax.scatter(targets_flat[:, dim_idx], predictions_flat[:, dim_idx], alpha=0.6, s=10)
            
            ax.plot([targets_flat[:, dim_idx].min(), targets_flat[:, dim_idx].max()], 
                    [targets_flat[:, dim_idx].min(), targets_flat[:, dim_idx].max()], 'r--', lw=2)
            ax.set_xlabel(f'Actual {coord_name}')
            ax.set_ylabel(f'Predicted {coord_name}')
            ax.set_title(f'Player {player} - {coord_name} Coordinate')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'regression_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regression prediction plots saved to: {save_path}")

def create_classification_plots(predictions, targets, output_dir, pov_team_sides=None):
    """Create plots for classification predictions."""
    # predictions are logits [N, num_places], targets are counts [N, num_places]
    
    # Convert logits to predicted counts for plotting
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    pred_probs = F.softmax(predictions_tensor, dim=-1)
    pred_counts = (pred_probs * 5.0).numpy()  # 5 agents total
    
    # Create scatter plot for place counts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enemy Location Prediction: Histogram Analysis', fontsize=16)
    
    # Overall scatter plot: predicted vs actual counts
    ax = axes[0, 0]
    if pov_team_sides is not None:
        ct_mask = pov_team_sides == 'CT'
        t_mask = pov_team_sides == 'T'
        
        if np.any(ct_mask):
            ax.scatter(targets[ct_mask].flatten(), pred_counts[ct_mask].flatten(), 
                        alpha=0.6, s=10, c='blue', label='CT', marker='o')
        if np.any(t_mask):
            ax.scatter(targets[t_mask].flatten(), pred_counts[t_mask].flatten(), 
                        alpha=0.6, s=10, c='red', label='T', marker='^')
        ax.legend()
    else:
        ax.scatter(targets.flatten(), pred_counts.flatten(), alpha=0.6, s=10)
        
    ax.plot([0, 5], [0, 5], 'r--', lw=2)
    ax.set_xlabel('Actual Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title('All Places: Predicted vs Actual Counts')
    ax.grid(True, alpha=0.3)
    
    # Distribution of actual vs predicted counts
    ax = axes[0, 1]
    if pov_team_sides is not None:
        ct_mask = pov_team_sides == 'CT'
        t_mask = pov_team_sides == 'T'
        
        if np.any(ct_mask):
            ax.hist(targets[ct_mask].flatten(), bins=np.arange(0, 6.5, 0.5), 
                    alpha=0.5, label='Actual (CT)', color='blue')
            ax.hist(pred_counts[ct_mask].flatten(), bins=np.arange(0, 6.5, 0.5), 
                    alpha=0.5, label='Predicted (CT)', color='lightblue')
        if np.any(t_mask):
            ax.hist(targets[t_mask].flatten(), bins=np.arange(0, 6.5, 0.5), 
                    alpha=0.5, label='Actual (T)', color='red')
            ax.hist(pred_counts[t_mask].flatten(), bins=np.arange(0, 6.5, 0.5), 
                    alpha=0.5, label='Predicted (T)', color='lightcoral')
    else:
        ax.hist(targets.flatten(), bins=np.arange(0, 6.5, 0.5), alpha=0.7, label='Actual')
        ax.hist(pred_counts.flatten(), bins=np.arange(0, 6.5, 0.5), alpha=0.7, label='Predicted')
    
    ax.set_xlabel('Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Place Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-place L1 count error
    ax = axes[1, 0]
    place_l1_error = np.mean(np.abs(pred_counts - targets), axis=0)
    # Ensure we have scalar values for plotting
    place_l1_error = np.asarray(place_l1_error).flatten()
    ax.bar(range(len(place_l1_error)), place_l1_error)
    ax.set_xlabel('Place Index')
    ax.set_ylabel('L1 Count Error')
    ax.set_title('L1 Count Error per Place')
    ax.grid(True, alpha=0.3)
    
    # Per-place exact accuracy
    ax = axes[1, 1]
    place_accuracy = []
    for i in range(pred_counts.shape[1]):  # num_places
        pred_place_counts = pred_counts[:, i]
        target_place_counts = targets[:, i]
        place_exact_matches = np.round(pred_place_counts) == target_place_counts
        place_accuracy.append(float(np.mean(place_exact_matches)))
    
    # Ensure we have scalar values for plotting
    place_accuracy = np.asarray(place_accuracy).flatten()
    ax.bar(range(len(place_accuracy)), place_accuracy)
    ax.set_xlabel('Place Index')
    ax.set_ylabel('Exact Match Accuracy')
    ax.set_title('Exact Match Accuracy per Place')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'classification_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification prediction plots saved to: {save_path}")


def _kde_plot(
    ax: Axes,
    x: list[float],
    y: list[float],
    size: int,
    cmap: str,
    alpha: float,
    alpha_range: list[float] | None,
    min_alpha: float,
    max_alpha: float,
    kde_lower_bound: float = 0.1,
    bandwidth: float | None = None,
) -> Axes:
    """Returns an `ax` with a kde plot."""
    # Calculate the kernel density estimate
    xy = np.vstack([x, y])
    kde = scipy.stats.gaussian_kde(xy)
    
    # Set custom bandwidth for sharper or smoother results
    if bandwidth is not None:
        kde.set_bandwidth(bandwidth)

    # Create a grid and evaluate the KDE on it
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xi, yi = np.mgrid[xmin : xmax : size * 1j, ymin : ymax : size * 1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # Set very low density values to NaN to make them transparent
    threshold = zi.max() * kde_lower_bound  # You can adjust this threshold
    zi[zi < threshold] = np.nan

    if alpha_range is not None:
        # Normalize KDE values
        zi_norm = zi / zi.max()

        # Create a color array with variable alpha
        colors = plt.cm.get_cmap(cmap)(zi_norm)
        colors[..., -1] = np.where(
            np.isnan(zi_norm),
            0,
            zi_norm * (max_alpha - min_alpha) + min_alpha,
        )
        _heatmap = ax.pcolormesh(xi, yi, zi, cmap=cmap, alpha=colors)
    else:
        _heatmap = ax.pcolormesh(xi, yi, zi, cmap=cmap, alpha=alpha)

    return ax


def verify_alpha_range(alpha_range: list[float]) -> list:
    """Verify that `alpha_range` is valid."""
    if len(alpha_range) != 2:
        msg = "alpha_range must have exactly 2 elements."
        raise ValueError(msg)
    min_val, max_val = alpha_range[0], alpha_range[1]
    if not (min_val >= 0 and min_val <= 1) or not (max_val >= 0 and max_val <= 1):
        msg = "alpha_range must have both values as floats between \
            0 and 1."
        raise ValueError(msg)
    if min_val > max_val:
        msg = "alpha_range[0] (min alpha) cannot be greater than \
            alpha[1] (max alpha)."
        raise ValueError(msg)
    return [min_val, max_val]

def heatmap(
    map_name: str,
    points: list[tuple[float, float, float]],
    size: int = 10,
    cmap: str = "RdYlGn",
    alpha: float = 0.5,
    *,
    alpha_range: list[float] | None = None,
    kde_lower_bound: float = 0.1,
    kde_bandwidth: float | None = None,
) -> tuple[Figure, Axes]:
    """Create a heatmap of points on a Counter-Strike map.

    Args:
        map_name (str): Name of the map to plot. E.g. "de_dust2"
            ("dust2" or "de_dust2.png" will not work).
        points (list[tuple[float, float, float]]): list of points to plot.
        size (int, optional): Size of the heatmap grid. Defaults to 10.
        cmap (str, optional): Colormap to use. Defaults to 'RdYlGn'.
        alpha (float, optional): Transparency of the heatmap. Defaults to 0.5.
        alpha_range (list[float, float], optional): When value is provided
            here,  points' transparency will vary based on the density, with
            min transparency of `alpha_range[0]` and max of `alpha_range[1]`.
            Defaults to `None`, meaning no variance of transparency.
        kde_lower_bound (float, optional): Lower bound for KDE density
            values. Defaults to 0.1.
        kde_bandwidth (float, optional): Bandwidth for KDE kernel. Lower values
            create sharper, more focused heatmaps. Higher values create smoother
            heatmaps. Defaults to None (uses scipy's automatic bandwidth).

    Raises:
        ValueError: Raises a ValueError if an invalid method is provided.

    Returns:
        tuple[Figure, Axes]: Matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=(1024 / 300, 1024 / 300), dpi=300)
    fig.patch.set_facecolor('white')  # Set background to white instead of black
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins

    image = f"{map_name}.png"
    map_is_lower = map_name.endswith("_lower")
    if map_is_lower:
        map_name = map_name.removesuffix("_lower")

    # Load and display the map
    map_img_path = awpy.data.MAPS_DIR / image
    if not map_img_path.exists():
        map_img_path_err = f"Map image not found: {map_img_path}. Might need to call `awpy get maps`"
        raise FileNotFoundError(map_img_path_err)

    map_bg = mpimg.imread(map_img_path)
    ax.imshow(map_bg, zorder=0, alpha=0.5)

    temp_points = points
    points = []
    warning = ""
    for point in temp_points:
        point_is_lower = awpy.plot.utils.is_position_on_lower_level(map_name, point)
        # If point is on same level as map, then keep, else ignore & warn.
        if point_is_lower == map_is_lower:
            points.append(point)
        else:
            warning = f"You are drawing on the {'lower' if map_is_lower else 'upper'} level of the map, but provided some points on the {'lower' if point_is_lower else 'upper'} level, which were ignored."  # noqa: E501
    if warning:
        warnings.warn(warning, UserWarning, stacklevel=2)

    x, y = [], []
    for point in points:
        x_point = awpy.plot.utils.game_to_pixel_axis(map_name, point[0], "x")
        y_point = awpy.plot.utils.game_to_pixel_axis(map_name, point[1], "y")

        # Check if the point is within bounds of the map image
        # if x_point < 0 or x_point > 1024 or y_point < 0 or y_point > 1024:
        #     continue

        x.append(x_point)
        y.append(y_point)

    # Check and/or set alpha_range
    min_alpha, max_alpha = 0, 1
    if alpha_range is not None:
        min_alpha, max_alpha = verify_alpha_range(alpha_range)

    ax = _kde_plot(
        ax,
        x,
        y,
        size,
        cmap,
        alpha,
        alpha_range,
        min_alpha,
        max_alpha,
        kde_lower_bound,
        kde_bandwidth,
    )

    ax.axis("off")
    
    return fig, ax


def create_prediction_heatmaps(predictions_list, targets_list, pov_team_sides_list, output_dir, map_name):
    """
    Create KDE heatmaps for multiple prediction instances with ground truth overlay.
    
    Args:
        predictions_list: List of prediction arrays, each of shape [num_predictions, 5, 2]
        targets_list: List of target arrays, each of shape [5, 2]
        pov_team_sides_list: List of team sides for each instance
        output_dir: Directory to save plots
        map_name: CS:GO map name for background
    """
    for idx, (predictions, target, pov_team_side) in enumerate(zip(predictions_list, targets_list, pov_team_sides_list)):
        # predictions: [num_predictions, 5, 2] - multiple predictions for 5 agents (X,Y only)
        # target: [5, 2] - ground truth for 5 agents (X,Y only)
        
        # Flatten all predictions to get all predicted points
        all_pred_points = predictions.reshape(-1, 2)  # [num_predictions * 5, 2]
        # Convert to list of tuples for heatmap function (add dummy Z=0)
        pred_points_tuples = [(point[0], point[1], 0) for point in all_pred_points]
        target_points_tuples = [(point[0], point[1], 0) for point in target]
        
        # Create heatmap for predictions
        fig, ax = heatmap(
            map_name=map_name,
            points=pred_points_tuples,
            size=120,  # Higher resolution
            cmap='plasma',
            alpha=0.7,
            kde_lower_bound=0.35,
            kde_bandwidth=0.15  # Adjust for smoother/sharper heatmaps
        )
        
        # Overlay ground truth points
        for i, point in enumerate(target_points_tuples):
            # Convert game coordinates to pixel coordinates
            x_pixel = awpy.plot.utils.game_to_pixel_axis(map_name, point[0], "x")
            y_pixel = awpy.plot.utils.game_to_pixel_axis(map_name, point[1], "y")
            
            ax.scatter(x_pixel, y_pixel, 
                        c='orange', s=20, marker='X', 
                        edgecolors='black', linewidth=1,
                        alpha=0.9, zorder=10,
                        label='Ground Truth' if i == 0 else "")
        
        # Add legend and title
        ax.legend(loc='upper right', framealpha=0.8, fontsize='small', markerscale=0.8)
        
        plt.title(f'Enemy Location Predictions - Sample {idx+1} ({pov_team_side})\n'
                    f'{predictions.shape[0]} predictions for 5 agents', 
                    fontsize=14, pad=20)
        
        # Save the plot
        save_path = output_dir / f'heatmap_sample_{idx+1}_{pov_team_side.lower()}.png'
        plt.savefig(save_path, dpi=300, pad_inches=0, facecolor='#100C07')
        plt.close()
        
        print(f"Heatmap saved to: {save_path}")


def create_prediction_heatmaps_grid(predictions_list, targets_list, pov_team_sides_list, 
                                  scaled_predictions_list, scaled_targets_list, output_dir, map_name):
    """
    Create a grid of KDE heatmaps with T samples on top row and CT samples on bottom row.
    
    Args:
        predictions_list: List of prediction arrays, each of shape [num_predictions, 5, 2] (unscaled, X,Y only)
        targets_list: List of target arrays, each of shape [5, 2] (unscaled, X,Y only)
        pov_team_sides_list: List of team sides for each instance
        scaled_predictions_list: List of scaled prediction tensors for Chamfer distance calculation
        scaled_targets_list: List of scaled target tensors for Chamfer distance calculation
        output_dir: Directory to save plots
        map_name: CS:GO map name for background
    """

    
    # Separate samples by team
    t_samples = []
    ct_samples = []
    
    for predictions, target, pov_team_side, scaled_pred, scaled_target in zip(
        predictions_list, targets_list, pov_team_sides_list, scaled_predictions_list, scaled_targets_list):
        if pov_team_side == 'T':
            t_samples.append((predictions, target, pov_team_side, scaled_pred, scaled_target))
        else:
            ct_samples.append((predictions, target, pov_team_side, scaled_pred, scaled_target))
    
    # Ensure we have 5 samples from each team
    t_samples = t_samples[:4]
    ct_samples = ct_samples[:4]
    
    # Create a 2x5 grid of subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#100C07')
    
    # Plot T samples on top row
    for col, (predictions, target, pov_team_side, scaled_pred, scaled_target) in enumerate(t_samples):
        ax = axes[0, col]
        
        # Calculate Chamfer distance using scaled coordinates
        chamfer_dist = chamfer_distance_batch(scaled_pred, scaled_target).item()
        
        # Flatten all predictions to get all predicted points (use unscaled for visualization)
        all_pred_points = predictions.reshape(-1, 3)  # [num_predictions * 5, 3]
        pred_points_tuples = [(point[0], point[1], point[2]) for point in all_pred_points]
        target_points_tuples = [(point[0], point[1], point[2]) for point in target]
        
        # Create individual heatmap for this subplot
        title = f'T-Side Sample {col+1}\nCD={chamfer_dist:.3f}'
        _create_single_heatmap_subplot(ax, map_name, pred_points_tuples, target_points_tuples, 
                                       title, predictions.shape[0])
    
    # Plot CT samples on bottom row
    for col, (predictions, target, pov_team_side, scaled_pred, scaled_target) in enumerate(ct_samples):
        ax = axes[1, col]
        
        # Calculate Chamfer distance using scaled coordinates
        chamfer_dist = chamfer_distance_batch(scaled_pred, scaled_target).item()
        
        # Flatten all predictions to get all predicted points (use unscaled for visualization)
        all_pred_points = predictions.reshape(-1, 3)  # [num_predictions * 5, 3]
        pred_points_tuples = [(point[0], point[1], point[2]) for point in all_pred_points]
        target_points_tuples = [(point[0], point[1], point[2]) for point in target]
        
        # Create individual heatmap for this subplot
        title = f'CT-Side Sample {col+1}\nCD={chamfer_dist:.3f}'
        _create_single_heatmap_subplot(ax, map_name, pred_points_tuples, target_points_tuples, 
                                       title, predictions.shape[0])
    
    # Adjust layout with no padding between subplots
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.subplots_adjust(left=-0.01, right=1.02, top=1, bottom=-0.05, hspace=-0.02, wspace=-0.02)
    
    # Save the grid plot
    save_path = output_dir / 'heatmap_grid_comparison.png'
    save_path_pdf = output_dir / 'heatmap_grid_comparison.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches=None, facecolor='#100C07')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches=None, facecolor='#100C07')
    plt.close()
    
    print(f"Grid heatmap saved to: {save_path}")


def _create_single_heatmap_subplot(ax, map_name, pred_points_tuples, target_points_tuples, title, num_predictions):
    """
    Helper function to create a single heatmap subplot.
    """
    # Load and display the map background
    image = f"{map_name}.png"
    map_img_path = awpy.data.MAPS_DIR / image
    if not map_img_path.exists():
        raise FileNotFoundError(f"Map image not found: {map_img_path}")
    
    map_bg = mpimg.imread(map_img_path)
    ax.imshow(map_bg, zorder=0, alpha=0.5)
    
    # Filter points that are within map bounds
    x, y = [], []
    for point in pred_points_tuples:
        x_point = awpy.plot.utils.game_to_pixel_axis(map_name, point[0], "x")
        y_point = awpy.plot.utils.game_to_pixel_axis(map_name, point[1], "y")
        x.append(x_point)
        y.append(y_point)
    
    # Create KDE plot if we have points
    if len(x) > 0 and len(y) > 0:
        ax = _kde_plot(
            ax, x, y, size=120, cmap='plasma', alpha=0.7,
            alpha_range=None, min_alpha=0, max_alpha=1,
            kde_lower_bound=0.35, bandwidth=0.2
        )
    
    # Overlay ground truth points
    for i, point in enumerate(target_points_tuples):
        x_pixel = awpy.plot.utils.game_to_pixel_axis(map_name, point[0], "x")
        y_pixel = awpy.plot.utils.game_to_pixel_axis(map_name, point[1], "y")
        
        ax.scatter(x_pixel, y_pixel, 
                   c='orange', s=75, marker='X', 
                   edgecolors='black', linewidth=1,
                   alpha=0.9, zorder=10)
    
    # Set title and remove axis (title now includes all info)
    # ax.set_title(title, fontsize=15, color='white', pad=0)
    ax.text(512, 2, title, fontsize=13, color='white', 
            verticalalignment='top', horizontalalignment='center')

    ax.axis('off')