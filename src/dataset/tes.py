from operator import gt
import pandas as pd
import geopandas as gpd
from mapminer import miners
import pyproj
from sklearn.base import defaultdict
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
from geotessera import GeoTessera

from src.utils import create_grid


logger = logging.getLogger(__name__)


class TESDataHandler:
    def __init__(self):
        
        # Initialize the client
        self.gt = GeoTessera()

    def read_data(self, filepath: str) -> gpd.GeoDataFrame:
        """
        Read geospatial data from a file and return a GeoDataFrame.
        """
        gdf_polygon = gpd.read_file(filepath)

        gdf_polygon = gdf_polygon.explode(index_parts=False).reset_index(drop=True)

        gdf_polygon = gdf_polygon.to_crs(4326)

        return gdf_polygon
    

    def get_tile(self, tile_lat, tile_lon, year=2024):
        embedding, crs, transform = self.gt.fetch_embedding(tile_lat, tile_lon, year=year)

        da = xr.DataArray(
                embedding,
                dims=("y", "x", 'band'),   # (C, H, W)
                coords={
                    "band": list(range(128)),
                },
                name=f"tile_{tile_lat}_{tile_lon}"
            )
        del embedding
        
        # Add CRS + transform
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)

        del crs, transform
        da = da.transpose('band','y', 'x')

        return da

    def tiles_to_download(self, polygon):
        tiles = self.gt.registry.load_blocks_for_region(polygon.bounds, year=2024)
        return tiles

    def tile_to_polygons(self, gdf_polygon: gpd.GeoDataFrame):
        _tile_to_polygons = defaultdict(list)
        for idx, polygon in enumerate(gdf_polygon.geometry):
            tiles = self.tiles_to_download(polygon)
            for lat, lon in tiles:
                _tile_to_polygons[(lat, lon)].append(polygon)
        return _tile_to_polygons
    

    def clip_to_geometry(self, polygon_embd: xr.Dataset, geometry: Polygon) -> xr.Dataset:
        project = pyproj.Transformer.from_crs("EPSG:4326", polygon_embd.rio.crs, always_xy=True).transform
        geom_projected = transform(project, geometry)

        clipped = polygon_embd.rio.clip(
            [geom_projected], crs=polygon_embd.rio.crs, drop=True
        )
            
        return clipped

    def get_dataframe(self, polygon_embd: xr.DataArray) -> pd.DataFrame:
        if polygon_embd is None:
            return pd.DataFrame()
        
        da = polygon_embd.stack(pixel=("y", "x"))
        da = da.transpose("pixel", "band")
        df = da.to_pandas() 
        df = df.reset_index(drop=True).dropna()
        del da
        return df
    
    def get_polygon_embd_from_tile(self, tile_lat, tile_lon, polygon_list, year):
        df = None
        da = self.get_tile(tile_lat, tile_lon, year)
        for polygon in polygon_list:

            da_clipped = self.clip_to_geometry(da, polygon)
            _df = self.get_dataframe(da_clipped)

            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df], ignore_index=True)
        return df

    def create_dataset_from_polygons_parquet(self,
                                             filepath: str,
                                             year: int,
                                             output_path: str,
                                             clip: bool,
                                             max_workers: int,
                                             cls: int = None,
                                             ):


        gdf_polygon = self.read_data(filepath)
        tile_to_polygons = self.tile_to_polygons(gdf_polygon)

        tasks = [(tile_lat, tile_lon, polygons, year) for (tile_lat, tile_lon), polygons in tile_to_polygons.items()]

        output_path = Path(output_path)
        if output_path.exists():
            logger.warning(f"{output_path} already exists. Overwriting.")
            output_path.unlink()

        writer = None
        schema = None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_polygon_embd_from_tile, *t): t for t in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Polygons"):
                res = future.result()
                
                if res.empty():
                    continue

                # Convert to Arrow Table
                table = pa.Table.from_pandas(res, preserve_index=False)

                if writer is None:  # first time, initialize
                    schema = table.schema
                    writer = pq.ParquetWriter(str(output_path), schema)

                writer.write_table(table)
                del res

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
        gdf_polygon = self.read_data(filepath)
        tile_to_polygons = self.tile_to_polygons(gdf_polygon)

        tasks = [(tile_lat, tile_lon, polygons, year) for (tile_lat, tile_lon), polygons in tile_to_polygons.items()]

        output_file = Path(output_path)
        if output_file.exists():
            logger.warning(f"{output_path} already exists. Overwriting.")
            output_file.unlink()
        writer = None
        schema = None

        with ThreadPoolExecutor(max_workers=max_workers) as executorm, open(output_path, 'a') as f_out:
            futures = {executorm.submit(self.get_polygon_embd_from_tile, *t): t for t in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Polygons"):
                res = future.result()
                if res.empty():
                    continue

                if cls:
                    res['class'] = cls

                res.to_csv(f_out, index=False, header=f_out.tell()==0)  # write header only once

        logging.info(f"Dataset saved to {output_path}")
        return pd.read_csv(output_path)  # final load to return dataframe

    def __repr__(self):
        return str("AEF Data Handler class")
