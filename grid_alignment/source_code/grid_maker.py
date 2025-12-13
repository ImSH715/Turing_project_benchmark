import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np

# ===============================
# INPUT
# ===============================

csv_path = "../Dataset/random_point.csv"
cell_size = 0.8  # grid cell size

# ===============================
# READ CSV
# ===============================

df = pd.read_csv(csv_path)

# x = lat, y = long (AS USER STATED)
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")
df = df.dropna(subset=["lat", "long"]).reset_index(drop=True)

if df.empty:
    raise ValueError("CSV has no valid coordinates.")

# ===============================
# CREATE POINT GEOMETRY
# ===============================

points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lat"], df["long"]),
    crs=None
)

# ===============================
# CREATE 3x3 GRID CENTERED ON POINT
# ===============================

records = []
geometries = []

half = cell_size / 2

for idx, row in points.iterrows():
    cx = row.geometry.x
    cy = row.geometry.y

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x_min = cx + dx * cell_size - half
            x_max = cx + dx * cell_size + half
            y_min = cy + dy * cell_size - half
            y_max = cy + dy * cell_size + half

            records.append({
                "point_id": idx,
                "dx": dx,
                "dy": dy
            })

            geometries.append(
                box(x_min, y_min, x_max, y_max)
            )

# ===============================
# CREATE GRID GDF
# ===============================

grid_gdf = gpd.GeoDataFrame(
    records,
    geometry=geometries,
    crs=None
)

# ===============================
# SAVE
# ===============================

grid_gdf.to_file("shapefile/grid_3x3_centered_cell_0_8.shp")

print("DONE: 3x3 grid created, point is exact center.")
