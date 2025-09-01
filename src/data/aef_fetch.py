import pandas as pd
import geopandas as gpd
from mapminer import miners
import pyproj
import xarray as xr
from shapely.ops import transform
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import gc
from tqdm import tqdm
import logging
import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


class AEFDataHandler:
    def __init__(self):
        self.miner = miners.GoogleEmbeddingMiner()

    def read_data(self, filepath: str) -> gpd.GeoDataFrame:
        df_forest = gpd.read_file(filepath)
        df_forest = df_forest.explode(index_parts=False).reset_index(drop=True)
        gdf_polygon = df_forest.to_crs(4326)
        return gdf_polygon

    def fetch_polygon_extent(self, polygon: Polygon, YEAR: int, clip: bool = True) -> xr.Dataset:
        try:
            polygon_embd = self.miner.fetch(
                polygon=polygon,
                daterange=f"{YEAR}-01-01/{YEAR+1}-08-19"
            )
            if clip:
                polygon_embd = self.clip_to_geometry(polygon_embd, polygon)
            return polygon_embd
        except Exception as e:
            logger.error(f"Error fetching polygon: {e}")
            return None
    
    def fetch_latlon_extent(self, lat:float, lon:float, radius:float, YEAR: int) -> xr.Dataset:
        try:
            polygon_embd = self.miner.fetch(
                lat=lat, lon=lon, radius=radius,
                daterange=f"{YEAR}-01-01/{YEAR+1}-08-19"
            )
            return polygon_embd.rename({'X': 'x', 'Y': 'y'} )
        except Exception as e:
            logger.error(f"Error fetching polygon: {e}")
            return None

    def clip_to_geometry(self, polygon_embd: xr.Dataset, geometry: Polygon) -> xr.Dataset:
        project = pyproj.Transformer.from_crs("EPSG:4326", polygon_embd.rio.crs, always_xy=True).transform
        geom_projected = transform(project, geometry)

        clipped = polygon_embd.rename({'X': 'x', 'Y': 'y'}).rio.clip(
            [geom_projected], crs=polygon_embd.rio.crs, drop=True
        )
        return clipped

    def get_dataframe(self, polygon_embd: xr.Dataset) -> pd.DataFrame:
        if polygon_embd is None:
            return pd.DataFrame()

        if polygon_embd.time.shape[0] > 1:
            df = polygon_embd.max(dim='time').to_dataframe().reset_index()
        else:
            df = polygon_embd.isel(time=-1).to_dataframe().reset_index()
            
        columns_to_drop = ['x', 'y']
        if 'time' in df.columns:
            columns_to_drop.append('time')

        if 'spatial_ref' in df.columns:
            columns_to_drop.append('spatial_ref')
            
        df.drop(columns=columns_to_drop, inplace=True)

        polygon_embd.close()
        gc.collect()
        return df

    def create_dataset_from_polygons_parquet(self,
                                    filepath: str,
                                    year: int,
                                    output_path: str,
                                    clip: bool,
                                    max_workers: int,
                                    cls: int = None,
                                    ) -> pd.DataFrame:
        """
        Process polygons and save results to a Parquet file incrementally.
        """
        gdf_polygon = self.read_data(filepath)

        tasks = [(gdf_polygon.iloc[i].geometry, year, clip) for i in range(len(gdf_polygon))]

        output_file = Path(output_path)
        if output_file.exists():
            logger.warning(f"{output_path} already exists. Overwriting.")
            output_file.unlink()

        writer = None
        schema = None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_polygon_extent, *t): t for t in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Polygons"):
                res = future.result()
                df = self.get_dataframe(res)
                if df.empty:
                    continue

                if cls:
                    df['class'] = cls

                # Convert to Arrow Table
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:  # first time, initialize
                    schema = table.schema
                    writer = pq.ParquetWriter(str(output_file), schema)

                writer.write_table(table)

        if writer:
            writer.close()

    def create_dataset_from_polygons_csv(self,
                                     filepath: str,
                                     year: int,
                                     output_path: str,
                                     clip: bool,
                                     max_workers: int,
                                     cls: int = None,
                                     ):
        """
        Process polygons efficiently, writing results incrementally.
        """
        df_forest = gpd.read_file(filepath)
        df_forest = df_forest.explode(index_parts=False).reset_index(drop=True)
        gdf_polygon = df_forest.to_crs(4326)

        tasks = [(gdf_polygon.iloc[i].geometry, year, clip) for i in range(len(gdf_polygon))]

        output_file = Path(output_path)
        if output_file.exists():
            logging.warning(f"{output_path} already exists. Appending results.")

        # Process in batches (better for memory management)
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_file, "a") as f_out:
            futures = {executor.submit(self.fetch_polygon_extent, *t): t for t in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Polygons"):
                res = future.result()
                df = self.get_dataframe(res)
                if df.empty:
                    continue

                if cls:
                    df['class'] = cls

                df.to_csv(f_out, index=False, header=f_out.tell()==0)  # write header only once

        logging.info(f"Dataset saved to {output_path}")
        return pd.read_csv(output_path)  # final load to return dataframe

    def __repr__(self):
        return str("AEF Data Handler class")
