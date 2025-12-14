import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

# --------------------------------------------------
# Config
# --------------------------------------------------
GRID_SHP = "shapefile/grid_3x3_centered_cell_0_8.shp"
DETECTION_SHP = "../Dataset/orange_trees.shp"
RANDOM_POINT_CSV = "../Dataset/random_point.csv"

OUTPUT_POINT_CSV = "../Dataset/final_center_points.csv"
OUTPUT_GRID_SHP = "../Dataset/final_3x3_grids.shp"

LIKELIHOOD_THRESHOLD = 0.5
MAX_SHIFT = 1

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def compute_likelihood(grid_geom, det_geom):
    inter = grid_geom.intersection(det_geom)
    if inter.is_empty:
        return 0.0
    return inter.area / grid_geom.area


def grid_center_from_cells(cells):
    minx, miny, maxx, maxy = cells.total_bounds
    return (minx + maxx) / 2, (miny + maxy) / 2


def shift_grid_to_point(grid_gdf, x, y):
    minx, miny, maxx, maxy = grid_gdf.total_bounds
    gx = (minx + maxx) / 2
    gy = (miny + maxy) / 2

    shifted = grid_gdf.geometry.translate(
        xoff=x - gx,
        yoff=y - gy
    )

    return gpd.GeoDataFrame(
        grid_gdf.drop(columns="geometry"),
        geometry=shifted,
        crs=grid_gdf.crs
    )

# --------------------------------------------------
# Load data
# --------------------------------------------------
grid_template = gpd.read_file(GRID_SHP)
det_gdf = gpd.read_file(DETECTION_SHP)
det_sindex = det_gdf.sindex
points_df = pd.read_csv(RANDOM_POINT_CSV)

final_points = []
final_grids = []

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for pid, row in tqdm(
    points_df.iterrows(),
    total=len(points_df),
    desc="Processing points"
):

    x = float(row["lat"])
    y = float(row["long"])

    grid_gdf = shift_grid_to_point(grid_template, x, y)
    success = False

    for shift_iter in range(MAX_SHIFT + 1):

        records = []

        for gi, grid_row in grid_gdf.iterrows():
            grid_geom = grid_row.geometry

            cand_idx = list(det_sindex.intersection(grid_geom.bounds))
            if not cand_idx:
                continue

            for di in cand_idx:
                det_geom = det_gdf.geometry.iloc[di]
                if not grid_geom.intersects(det_geom):
                    continue

                records.append({
                    "grid_id": gi,
                    "det_id": di,
                    "likelihood": compute_likelihood(grid_geom, det_geom)
                })

        if not records:
            break

        df = pd.DataFrame(records)

        high_cells = df[df["likelihood"] >= LIKELIHOOD_THRESHOLD]

        if high_cells.empty:
            break

        # Count per detection
        det_summary = (
            high_cells.groupby("det_id")
            .size()
            .reset_index(name="count")
        )

        best_det = det_summary.sort_values(
            "count", ascending=False
        ).iloc[0]

        det_id = best_det["det_id"]
        cell_count = best_det["count"]

        det_geom = det_gdf.geometry.iloc[det_id]

        # Case: multiple cells -> success
        if cell_count >= 2:
            selected_cells = grid_gdf.loc[
                high_cells[
                    high_cells["det_id"] == det_id
                ]["grid_id"]
            ]

            cx, cy = grid_center_from_cells(selected_cells)

            final_points.append({
                "point_id": pid,
                "lat": cx,
                "long": cy
            })

            success = True
            break

        # Case: single cell -> shift grid
        if cell_count == 1 and shift_iter < MAX_SHIFT:
            c = det_geom.centroid
            grid_gdf = shift_grid_to_point(
                grid_template,
                c.x,
                c.y
            )
            continue

    if success:
        grid_gdf["point_id"] = pid
        final_grids.append(grid_gdf)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
if final_points:
    pd.DataFrame(final_points).to_csv(
        OUTPUT_POINT_CSV, index=False
    )

if final_grids:
    gpd.GeoDataFrame(
        pd.concat(final_grids, ignore_index=True),
        crs=grid_template.crs
    ).to_file(OUTPUT_GRID_SHP)

print(f"Total input points: {len(points_df)}")
print(f"Final center points: {len(final_points)}")
print(f"Final grids: {len(final_grids)}")
