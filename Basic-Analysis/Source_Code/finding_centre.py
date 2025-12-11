import geopandas as gpd
import shapefile as shp
import random
import rasterio

gdf = gpd.read_file(r"Z:\ai4eo\Shared\2025_Turing_L\datasets\orange_trees\orange_trees.shp")
sf = shp.Reader(r"C:\Users\naya0\Uni\COM Turing Project\Turing-Project\Basic-Analysis\Source_Code\tree_centroid.shp")

#gdf_centroid = gdf.copy()
#gdf_centroid['geometry'] = gdf.geometry.centroid

#gdf_centroid.to_file("tree_centroid.shp")

shapes = sf.shapes()
print(shapes.points)

random_shapes = []

for dx, dy in shapes:
    random_shapes = shapes[random(dx), random(dy)]

print(random_shapes)