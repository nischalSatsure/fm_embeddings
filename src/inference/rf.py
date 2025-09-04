import os
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rioxarray
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from pathlib import Path
from ..dataset.aef import AEFDataHandler
import dask.array as da
import logging

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

class AEFPredictor:
    """
    A class to handle polygon-based predictions and evaluation against reference data.
    """

    def __init__(self, model, year, output_dir="predictions"):
        self.model = model
        self.year = year
        self.output_dir = output_dir
        self.datahandler = AEFDataHandler()

        os.makedirs(self.output_dir, exist_ok=True)

    # def _get_polygon(self, polygon_path):

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


            # Wrap predictions in Dask (lazy)
            preds_dask = da.from_array(
                preds.reshape(roi.x.size, roi.y.size),
                chunks=("auto", "auto")  # let Dask decide
            )

            # Add predictions as new variable
            roi["preds"] = (("x", "y"), preds_dask)

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

    def multithreaded_predictions(
        self,
        gdf_grids,
        max_workers=4,
    ):
        """
        Run predictions on polygons in batches with multi-threading.
        """

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
                             eval_raster,
                             inference_region,
                             max_workers=4,
                             target_lulc_value=8,
                             grid_size=5000,
                             grid_overlap=500,
                             min_area=1e8,
                             no_grids=None):
        """
        Evaluate predicted grids against reference LULC data.
        If results are file paths, will load them before evaluation.
        """
        evaluator = rioxarray.open_rasterio(eval_raster)

        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        conf_matrices = []

        gdf_grids = self.datahandler.read_data_with_grids(inference_region, min_area=min_area, grid_size=grid_size, grid_overlap=grid_overlap)

        if no_grids:
            gdf_grids = gdf_grids.iloc[random.sample(range(len(gdf_grids)), no_grids)]

        for grid in tqdm(self.multithreaded_predictions(gdf_grids,
                                              max_workers=max_workers),
                         total=no_grids if no_grids else None,
                         desc="Evaluating"):

            if grid.rio.crs != evaluator.rio.crs:
                grid = grid.rio.reproject(evaluator.rio.crs)

            bbox = grid.rio.bounds()
            big_clip = evaluator.rio.clip_box(*bbox)

            small_aligned = grid.rio.reproject_match(big_clip)
            # small_aligned = grid.clip(min=0, max=1)
            small_aligned = small_aligned.where((small_aligned != -9223372036854775808) & (small_aligned != 255.), np.nan)


            # Flatten
            small_flat = small_aligned.values.flatten()
            big_flat = big_clip.values.flatten()

        
            mask = ~np.isnan(small_flat) & ~np.isnan(big_flat)
            small_flat = small_flat[mask]
            big_flat = big_flat[mask]
            
            # Binarize ground truth
            big_flat_binary = (big_flat == target_lulc_value).astype(int)

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
                        inference_region,
                        output_path,
                        max_workers=4,
                        grid_size=5000,
                        grid_overlap=500,
                        min_area=1e8,
                        clip=True,
                        no_grids=None,
                        chunks={"x": 2048, "y": 2048}):
        """
        Save forest cover predictions to raster without blowing up RAM.
        """
        # Collect predictions as xarrays (preferably already dask-backed)

        gdf_grids = self.datahandler.read_data_with_grids(inference_region, 
                                                          min_area=min_area, 
                                                          grid_size=grid_size, 
                                                          grid_overlap=grid_overlap)
        if no_grids:
            gdf_grids = gdf_grids.iloc[random.sample(range(len(gdf_grids)), no_grids)]

        predictions = []
        for pred in self.multithreaded_predictions(gdf_grids, max_workers=max_workers):
            if pred is not None:
                # Ensure dask-backed and rechunk
                arr = pred.chunk(chunks)
                predictions.append(arr)

        # Merge lazily with dask
        merged = rioxarray.merge.merge_arrays(predictions)

        # Rechunk to control memory footprint during write
        merged = merged.chunk(chunks)

        if clip: 
            boundary = self.datahandler.read_data(inference_region).union_all() # to get the single polygon 
            merged = self.datahandler.clip_to_geometry(merged, boundary)
            # merged = merged.clip(min=0, max=1) # clipping involved reprojection that sometimes creates null values.

        # Write out lazily, dask handles chunk-wise writing
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            merged.rio.to_raster(output_path,  windowed=True, tiled=True, BIGTIFF="IF_SAFER")
        else:
            merged.rio.to_raster("prediction.tif", windowed=True, tiled=True, BIGTIFF="IF_SAFER")

    def visualize_latlon(self, lat: float, lon: float, radius: float, alpha=0.4, save_tiff=False):
        import hvplot.xarray

        roi = self._predict_single_latlon(lat, lon, radius)
        # print(roi)
        prediction_layer = roi.rio.reproject(3857)

        # clipping values which reprojecting introduces
        # prediction_layer = prediction_layer.clip(min=0)

        class_colors = {
                        0: "saddlebrown",  # background / no data / buildup / agriland
                        1: "forestgreen",  # forest 
                    }

        # Create the image plot
        prediction_plot = prediction_layer.hvplot.image(
            cmap=list(class_colors.values()),  # Choose a colormap for predictions
            alpha=alpha,
            width=700,
            height=600,
            title='Forest Cover Prediction',
            tiles='EsriImagery',
            project=True,
            clim=(0,1),
        )

        if save_tiff:
            prediction_layer.rio.to_raster('prediction.tif')

        return prediction_plot
    
if __name__ == "__main__":
    import joblib
    import hvplot.xarray

    model_path = "experiments/rf_aef_v1/rf_aef_v1_model.pkl"
    year = 2024

    model = joblib.load(model_path)
    handler = AEFPredictor(model, 2024)

    # handler.visualize_latlon(24.06101, 74.60881, 100)   



