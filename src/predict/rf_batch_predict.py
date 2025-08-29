import os
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rioxarray
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from pathlib import Path
from ..data.aef_fetch import AEFDataHandler
from shapely.geometry import Polygon
import logging

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

class AEFPredictor:
    """
    A class to handle polygon-based predictions and evaluation against reference data.
    """

    def __init__(self, model, year, output_dir="predictions", target_lulc_value=8):
        self.model = model
        self.year = year
        self.output_dir = output_dir
        self.datahandler = AEFDataHandler()
        self.target_lulc_value = target_lulc_value

        os.makedirs(self.output_dir, exist_ok=True)

    def _predict_single_polygon(self, polygon):
        """
        RAM-efficient single polygon prediction.
        """
        try:
            roi = self.datahandler.fetch_polygon_extent(polygon, self.year)
            df_test = self.datahandler.get_dataframe(roi)

            preds = self.model.predict(df_test)

            # Clear df_test
            del df_test
            gc.collect()

            # Add predictions as new variable
            roi["preds"] = (("x", "y"), preds.reshape(roi.x.size, roi.y.size))

            del preds
            gc.collect()

            prediction_layer = roi.preds.rio.write_crs(roi.rio.crs)
            prediction_layer = prediction_layer.clip(min=0, max=1).astype(np.uint8)

            return prediction_layer.transpose('y', 'x')

        except Exception as e:
            print(f"Error processing polygon: {e}")
            gc.collect()
            return None
        
    def _predict_single_latlon(self, 
                                lat: float, 
                                lon: float, 
                                radius: float
                                ):
        """
        RAM-efficient single polygon prediction.
        """
        try:
            roi = self.datahandler.fetch_latlon_extent(lat, lon, radius, self.year)
            df_test = self.datahandler.get_dataframe(roi)

            preds = self.model.predict(df_test)

            # Clear df_test
            del df_test
            gc.collect()

            # Add predictions as new variable
            roi["preds"] = (("x", "y"), preds.reshape(roi.x.size, roi.y.size))

            del preds
            gc.collect()

            prediction_layer = roi.preds.rio.write_crs(roi.rio.crs)
            prediction_layer = prediction_layer.clip(min=0, max=1).astype(np.uint8)

            return prediction_layer.transpose('y', 'x')

        except Exception as e:
            print(f"Error processing polygon: {e}")
            gc.collect()
            return None

    def run_predictions(
        self,
        gdf_inference_path,
        max_workers=4,
        no_grids=None,
    ):
        """
        Run predictions on polygons in batches with multi-threading.
        """
        gdf_grids = self.datahandler.read_data(gdf_inference_path)

        if no_grids:
            gdf_grids = gdf_grids.iloc[random.sample(range(len(gdf_grids)), no_grids)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._predict_single_polygon, polygon): idx
                for idx, polygon in enumerate(gdf_grids.geometry)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
                try:
                    result = future.result()
                    if result is not None:
                        yield result   # stream result immediately
                except Exception as e:
                    print(f"Task failed: {e}")

    def evaluate_predictions(self,
                             datapath,
                             gdf_inference_path,
                             max_workers=4,
                             no_grids=None):
        """
        Evaluate predicted grids against reference LULC data.
        If results are file paths, will load them before evaluation.
        """
        gsc_lulc = rioxarray.open_rasterio(datapath)

        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        conf_matrices = []

        for grid in tqdm(self.run_predictions(gdf_inference_path,
                                              max_workers=max_workers,
                                              no_grids=no_grids),
                         total=no_grids if no_grids else None,
                         desc="Evaluating"):

            if grid.rio.crs != gsc_lulc.rio.crs:
                grid = grid.rio.reproject(gsc_lulc.rio.crs)

            bbox = grid.rio.bounds()
            big_clip = gsc_lulc.rio.clip_box(*bbox)

            small_aligned = grid.rio.reproject_match(big_clip)
            # small_aligned = grid.clip(min=0, max=1)
            small_aligned = small_aligned.where(small_aligned != -9223372036854775808, np.nan)


            # Flatten
            small_flat = small_aligned.values.flatten()
            big_flat = big_clip.values.flatten()

        
            mask = ~np.isnan(small_flat) & ~np.isnan(big_flat)
            small_flat = small_flat[mask]
            big_flat = big_flat[mask]
            
            # Binarize ground truth
            big_flat_binary = (big_flat == self.target_lulc_value).astype(int)

            # Metrics
            precision_scores.append(precision_score(big_flat_binary, small_flat, zero_division=0))
            recall_scores.append(recall_score(big_flat_binary, small_flat, zero_division=0))
            accuracy_scores.append(accuracy_score(big_flat_binary, small_flat))
            conf_matrices.append(confusion_matrix(big_flat_binary, small_flat))

        return {
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
            "accuracy": float(np.mean(accuracy_scores)),
        }

    def save_forest_cover(self,
                        output_path,
                        gdf_inference_path,
                        max_workers=4,
                        no_grids=None,
                        chunks={"x": 2048, "y": 2048}):
        """
        Save forest cover predictions to raster without blowing up RAM.
        """
        # Collect predictions as xarrays (preferably already dask-backed)
        predictions = list(self.run_predictions(
            gdf_inference_path,
            max_workers=max_workers,
            no_grids=no_grids
        ))

        # Merge lazily with dask
        merged = rioxarray.merge.merge_arrays(predictions)

        # Rechunk to control memory footprint during write
        merged = merged.chunk(chunks)

        # Write out lazily, dask handles chunk-wise writing
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            merged.rio.to_raster(output_path, tiled=True, BIGTIFF="IF_SAFER")

        else:
            merged.rio.to_raster("prediction.tif", tiled=True, BIGTIFF="IF_SAFER")

    def visualize_latlon(self, lat: float, lon: float, radius: float, save_tiff=False):

        roi = self.datahandler.fetch_latlon_extent(lat, lon, radius, self.year)
        prediction_layer = roi.rio.reproject(3857)

        # clipping values which reprojecting introduces
        prediction_layer = prediction_layer.clip(min=0)

        # Create the image plot
        prediction_plot = prediction_layer.hvplot.image(
            cmap='viridis',  # Choose a colormap for predictions
            alpha=0.6,
            width=700,
            height=600,
            title='Forest Cover Prediction',
            tiles='EsriImagery',
            project=True,
            clim=(prediction_layer.min().item(), prediction_layer.max().item()),
        )

        if save_tiff:
            prediction_layer.rio.to_raster('prediction.tif')

        return prediction_plot