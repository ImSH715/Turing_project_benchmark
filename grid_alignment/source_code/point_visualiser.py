import rasterio
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# loading .tif
tif_path = "../Dataset/Orange_trees.tif"
with rasterio.open(tif_path) as src:
    raster = src.read(1)
    extent = [
        src.bounds.left,
        src.bounds.right,
        src.bounds.bottom,
        src.bounds.top
    ]
    crs = src.crs

#loading csv file
df = pd.read_csv("../Dataset/random_point.csv")

# lat = x, long = y
geometry = [Point(xy) for xy in zip(df['lat'], df['long'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

# Visulalisation
plt.figure(figsize=(8, 8))
plt.imshow(raster, extent=extent, cmap="gray")
gdf.plot(ax=plt.gca(), color="red", markersize=10)
plt.title("Orange Trees with Random Points")
plt.show()
