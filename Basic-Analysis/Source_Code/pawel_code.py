import geopandas as gpd
import random
import rasterio


df = gpd.read_file(
    r"Z:\ai4eo\Shared\2025_Turing_L\datasets\orange_trees\orange_trees.shp")
tif = r"Z:\ai4eo\Shared\2025_Turing_L\datasets\orange_trees\Orange_trees.tif"

raster = rasterio.open(tif)
df['geometry'] = df.geometry.centroid

df.to_file("tree_centroid.shp")

rf = gpd.GeoDataFrame.copy(df)

bounds  = raster.bounds
rf.geometry = df.geometry.translate(
    xoff=random.uniform(-150.000, 150.000), yoff=random.uniform(-150.000, 150.000))

# for p in df.itertuples():
#     p[0] + random.uniform(-150.000, 150.000)
#     p[1] + random.uniform(-150.000, 150.000)
#     rf.
rf.to_file("random_centroid.shp")
