import geopandas as gpd
from shapely.geometry import box

def create_grid(polygon, grid_size=5000, overlap=100):
    """
    Create a grid of squares over a polygon with overlap.
    
    polygon   : shapely.geometry.Polygon
    grid_size : size of grid cells in meters (default 5000 = 5km)
    overlap   : overlap between adjacent cells in meters
    """
    minx, miny, maxx, maxy = polygon.bounds

    step = grid_size - overlap   # move less than full cell size to get overlap
    grid_cells = []
    
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
            y += step
        x += step
    
    # Clip to polygon (keep only intersecting parts)
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:3857")
    grid = grid[grid.intersects(polygon)]
    
    return grid 