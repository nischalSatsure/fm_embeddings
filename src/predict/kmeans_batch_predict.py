from src.data.aef_fetch import AEFDataHandler
import os
from shapely.geometry import box
import geopandas as gpd
import gc

def fetch_predict_polygon(
        model, task_id, YEAR, 
        polygon=None,

        save_results=True, output_dir="predictions"):
    """
    More RAM-efficient version that optionally saves results to disk
    """
    try:
        
        datahandler.

        # Make predictions

        preds = model.predict(df_test)

        # Clear df_test immediately after use
        del df_test
        gc.collect()

        # Process predictions efficiently
        
        cluster_mask = np.isin(preds, [0,2,4,5,7,8,9,10,11,13,15,16,17])
        preds = cluster_mask.astype(np.uint8)  # Use uint8 instead of int64

        # Create prediction layer
        test_area['preds'] = (('X', 'Y'), preds.reshape(test_area.X.size, test_area.Y.size))

        # Clear preds array
        del preds, cluster_mask

        prediction_layer = test_area.preds.rio.write_crs(test_area.rio.crs)
        prediction_layer = prediction_layer.clip(min=0, max=1)

        # Clear test_area
        del test_area

        if save_results:
            # Save to disk instead of keeping in memory

            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/prediction_{task_id:04d}.nc"
            prediction_layer.to_netcdf(filename)

            # Return filename instead of data
            del prediction_layer
            gc.collect()
            return filename
        else:
            # Return the actual layer (use this option carefully)
            return prediction_layer
        
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        gc.collect()  # Clean up on error too
        return None
        
def run_predictions_batched(gdf_inference_path, model, batch_size=10, max_workers=4):
    """
    Moderate RAM usage: processes in batches
    """
    datahandler = AEFDataHandler()
    gdf_grids = datahandler.read_data(gdf_inference_path)


    all_results = []

    total_batches = (len(gdf_grids) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(gdf_grids), batch_size):
        print(f"Processing batch {batch_idx//batch_size + 1}/{total_batches}")

        batch_end = min(batch_idx + batch_size, len(gdf_grids))
        batch_tasks = [
            (model, google_miner, gdf_grids.iloc[i].geometry, i)
            for i in range(batch_idx, batch_end)
        ]

        batch_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(fetch_predict_efficient, t, save_results=False): i
                for i, t in enumerate(batch_tasks)
            }

            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")

        # Process batch results (e.g., merge, save, etc.)
        all_results.extend(batch_results)

        # Clear batch results and force garbage collection
        del batch_results
        gc.collect()

        print(f"Batch {batch_idx//batch_size + 1} completed. Total results: {len(all_results)}")

    return all_results