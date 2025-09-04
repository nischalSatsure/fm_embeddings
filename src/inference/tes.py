from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rioxarray
from tqdm import tqdm
from pathlib import Path
from ..dataset.tes import TESDataHandler
import dask.array as da
import logging
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TESPredictor:
    def __init__(self, model):
        self.datahandler = TESDataHandler()
        self.model = model

    def _predict_single_tile(self, tile_lat, tile_lon, year):
        """
        Run prediction on a single tile.
        """
        # Prepare the input data for the model
        tile = self.datahandler.get_tile(tile_lat, tile_lon, year)

        df = self.datahandler.get_dataframe(tile)

        # Run the model prediction
        prediction = self.model.predict(df)

        preds_dask = da.from_array(
                prediction.reshape(tile.x.size, tile.y.size),
                chunks=("auto", "auto")  # let Dask decide
            )

        del prediction, df


        # Post-process the prediction
        tile["preds"] = (("x", "y"), preds_dask)
        prediction = tile["preds"].rio.write_crs(tile.rio.crs)
        prediction = prediction.clip(min=0, max=1).astype(np.uint8)

        return prediction.transpose('y', 'x')

    def multithreaded_predictions(
        self,
        tiles,
        max_workers=4,
    ):
        """
        Run predictions on tiles in batches with multi-threading.
        """

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._predict_single_tile, tile_lat, tile_lon, year): idx
                for idx, (tile_lat, tile_lon, year) in enumerate(tiles)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
                try:
                    result = future.result()
                    if result is not None:
                        yield result   # stream result immediately
                except Exception as e:
                    print(f"Task failed: {e}")

    def save_predictions(self,
                        inference_region,
                        output_path,
                        year,
                        max_workers=4,
                        clip=True,
                        chunks={"x": 2048, "y": 2048}):
        """
        Save forest cover predictions to raster without blowing up RAM.
        """
        # Collect predictions as xarrays (preferably already dask-backed)

        gdf_grids = self.datahandler.read_data(inference_region)

        tiles_to_load = self.datahandler.tiles_to_download(gdf_grids)

        tasks = [(tile_lat, tile_lon, year) for (tile_lat, tile_lon) in tiles_to_load]

        predictions = []
        for pred in self.multithreaded_predictions(tasks, max_workers=max_workers):
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

    def evaluate_through_dataframe(self, df):
        """Evaluate the model using a DataFrame."""
        X = df.drop(columns=["class_"])
        y = df["class_"]
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred)
