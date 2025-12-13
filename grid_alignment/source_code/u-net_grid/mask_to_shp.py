import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

MASK_PATH = "Dataset/output/mask_full.tif"
OUT_SHP = "Dataset/output/tree_crown.shp"
THRESH = 0.5

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    transform = src.transform
    crs = src.crs

binary = mask > THRESH

polys = []
for geom, val in shapes(binary.astype("uint8"), transform=transform):
    if val == 1:
        polys.append(shape(geom))

gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
gdf.to_file(OUT_SHP)

print("Saved shapefile:", OUT_SHP)
