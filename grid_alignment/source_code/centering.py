import geopandas as gpd
import pandas as pd

# File paths
grid_shp = "shapefile/grid_3x3_centered_cell_0_8.shp"
detections_shp = "../Dataset/detections.shp"
output_csv = "../Dataset/overlap_clown_center.csv"

# Load data
grid_gdf = gpd.read_file(grid_shp)
detections_gdf = gpd.read_file(detections_shp)

# Spatial join: grid × detection overlap
overlaps = gpd.sjoin(
    detections_gdf,
    grid_gdf,
    how="inner",
    predicate="intersects"
)

# Compute detection center for each overlap
overlaps["center"] = overlaps.geometry.representative_point()

# Extract coordinates
overlaps["center_long"] = overlaps["center"].x
overlaps["center_lat"] = overlaps["center"].y

# Build output table
output_df = overlaps[[
    "center_long",
    "center_lat"
]].copy()

# Save CSV
output_df.to_csv(output_csv, index=False)
