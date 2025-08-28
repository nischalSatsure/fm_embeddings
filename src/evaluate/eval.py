from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray
import rioxarray
import geopandas as gpd
# Assuming a value of 1 in gsc_lulc represents the target class (e.g., forest)


def grids_results_against_data(results, datapath):
    target_lulc_value = 8

    gsc_lulc = rioxarray.open_rasterio(datapath)

    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    conf_matrices = []


    for grid in results:
        grid = grid.rename({'X':'x','Y':'y'}).transpose('y', 'x')

        if grid.rio.crs != gsc_lulc.rio.crs:
            grid = grid.rio.reproject(gsc_lulc.rio.crs)
        bbox = grid.rio.bounds()

        # Clip big raster to that bounding box
        big_clip = gsc_lulc.rio.clip_box(*bbox)

        # Reproject and match the smaller grid to the clipped larger raster
        small_aligned = grid.rio.reproject_match(big_clip)

        # Flatten the arrays and remove NaNs
        small_flat = small_aligned.values.flatten()
        big_flat = big_clip.values.flatten()

        # Remove NaN values
        mask = ~np.isnan(small_flat) & ~np.isnan(big_flat)
        small_flat = small_flat[mask]
        big_flat = big_flat[mask]

        # Binarize big_flat based on the target_lulc_value
        big_flat_binary = (big_flat == target_lulc_value).astype(int)

        # Calculate metrics
        precision = precision_score(big_flat_binary, small_flat)
        recall = recall_score(big_flat_binary, small_flat)
        accuracy = accuracy_score(big_flat_binary, small_flat)
        conf_matrix = confusion_matrix(big_flat_binary, small_flat)

        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        conf_matrices.append(conf_matrix)

    precision_scores = np.asarray(precision_scores).mean()
    recall_scores = np.asarray(recall_scores).mean()
    accuracy_scores = np.asarray(accuracy_scores).mean()

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "accuracy": accuracy_scores,
    }
