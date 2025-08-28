import os
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rioxarray
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from src.data.aef_fetch import AEFDataHandler


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

    def _predict_single_polygon(self, polygon, save_results=False, idx=None):
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
            prediction_layer = prediction_layer.clip(min=0, max=1)

            if save_results:
                filename = os.path.join(
                    self.output_dir,
                    f"prediction_{idx}.nc" if idx is not None else "prediction.nc"
                )
                prediction_layer.to_netcdf(filename)
                del prediction_layer
                gc.collect()
                return filename
            else:
                return prediction_layer

        except Exception as e:
            print(f"Error processing polygon: {e}")
            gc.collect()
            return None

    def run_predictions(
        self,
        gdf_inference_path,
        batch_size=10,
        max_workers=4,
        no_grids=None,
        save_results=False
    ):
        """
        Run predictions on polygons in batches with multi-threading.
        Returns either a list of file paths (if save_results=True) or a list of in-memory grids.
        """
        gdf_grids = self.datahandler.read_data(gdf_inference_path)

        if no_grids:
            gdf_grids = gdf_grids.iloc[random.sample(range(len(gdf_grids)), no_grids)]

        all_results = []
        total_batches = (len(gdf_grids) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(gdf_grids), batch_size):
            batch_end = min(batch_idx + batch_size, len(gdf_grids))
            batch_tasks = gdf_grids.iloc[batch_idx:batch_end].geometry.tolist()

            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._predict_single_polygon,
                        polygon,
                        save_results,
                        batch_idx + i
                    ): i
                    for i, polygon in enumerate(batch_tasks)
                }

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Batch {batch_idx//batch_size + 1}/{total_batches}"):
                    try:
                        result = future.result()
                        if result is not None:
                            batch_results.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")

            all_results.extend(batch_results)

            # Clean batch
            del batch_results
            gc.collect()

        return all_results

    def evaluate_predictions(self, results, datapath):
        """
        Evaluate predicted grids against reference LULC data.
        If results are file paths, will load them before evaluation.
        """
        gsc_lulc = rioxarray.open_rasterio(datapath)

        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        conf_matrices = []

        for grid in tqdm(results, desc="Evaluating predictions"):
            if isinstance(grid, str):  # file path
                grid = rioxarray.open_rasterio(grid)

            # Ensure consistent dimensions
            grid = grid.transpose('y', 'x')

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
