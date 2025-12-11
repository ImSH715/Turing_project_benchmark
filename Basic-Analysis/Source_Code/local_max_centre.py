import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.ndimage import maximum_filter
from shapely.geometry import Point
from rasterio.mask import mask

def local_centroid_per_row(chm_tif, crowns_shp, input_csv, output_csv, filter_size=7):

    print("Load CHM…")
    with rasterio.open(chm_tif) as src:
        src_crs = src.crs
        src_nodata = src.nodata

    print("Load crown polygons…")
    crowns = gpd.read_file(crowns_shp)
    
    # CRS 통일 (Raster 기준)
    if crowns.crs != src_crs:
        crowns = crowns.to_crs(src_crs)

    print("Load input CSV…")
    df = pd.read_csv(input_csv)
    print("Total input rows:", len(df))

    results = []
    
    # 미리 Centroid 계산 (속도 향상)
    crown_centroids = crowns.geometry.centroid

    # Raster 핸들러를 루프 밖에서 엽니다 (반복적 open 방지)
    with rasterio.open(chm_tif) as src:
        
        for i, row in df.iterrows():
            # -------------------------------------------------------
            # [중요] CSV의 어떤 컬럼이 X(Easting)이고 Y(Northing)인지 확인 필요
            # 로그상의 수치(700만/60만)를 보아 lat이 Y, long이 X라고 가정하고 매핑합니다.
            # 만약 CSV 헤더가 반대라면 row["long"], row["lat"] 순서로 넣으세요.
            # Point(x, y) 순서여야 합니다.
            # -------------------------------------------------------
            try:
                # 일반적인 경우: Longitude=X, Latitude=Y
                pt_x = float(row["long"])
                pt_y = float(row["lat"])
                
                # 좌표값 크기로 보정 (로그 기반 추측: 700만이 Y, 60만이 X여야 함)
                # 입력값이 반대로 들어가 있다면 아래 주석을 해제하여 스왑하세요.
                if pt_x > pt_y: 
                     pt_x, pt_y = pt_y, pt_x
                
                pt = Point(pt_x, pt_y)
            except Exception as e:
                print(f"Row {i}: Coordinate error - {e}")
                continue

            # 1. 가장 가까운 Crown 찾기
            dists = crown_centroids.distance(pt)
            nearest_crown_index = dists.idxmin()
            
            # 여기서 선택된 crown의 geometry
            crown_geom = crowns.loc[nearest_crown_index].geometry
            centroid = crown_geom.centroid

            # 2. CHM Crop (핵심 수정 부분)
            try:
                # mask는 (cropped_image, cropped_transform)을 반환합니다.
                # 반드시 out_transform을 받아야 합니다.
                crop, out_transform = mask(src, [crown_geom], crop=True)
            except ValueError:
                # Crown이 이미지 범위를 벗어난 경우 등
                continue

            crop = crop[0] # 첫번째 밴드
            
            # NoData 처리
            if src_nodata is not None:
                crop = np.nan_to_num(crop, nan=src_nodata)
                crop_temp = np.where(crop == src_nodata, -9999, crop)
            else:
                crop_temp = np.nan_to_num(crop, nan=-9999)

            # 3. Local Maxima 찾기
            lmax = maximum_filter(crop_temp, size=filter_size)
            
            # Local max이면서 0보다 큰 값 (나무 높이)
            is_top = (crop_temp == lmax) & (crop_temp > 0)

            if not np.any(is_top):
                # Maxima를 못 찾은 경우 (crown 중심이라도 반환할지 결정)
                results.append({
                    "input_index": i,
                    "polygon_index": int(nearest_crown_index),
                    "tree_top_x": np.nan,
                    "tree_top_y": np.nan,
                    "tree_height": 0,
                    "centre_x": centroid.x,
                    "centre_y": centroid.y
                })
                continue

            # 4. 가장 높은 픽셀 좌표 추출
            r_indices, c_indices = np.where(is_top)
            
            # 여러 maxima 중 가장 높은 것 하나 선택
            heights = crop[r_indices, c_indices]
            maxpos = heights.argmax()
            
            r = r_indices[maxpos]
            c = c_indices[maxpos]
            max_height = float(heights[maxpos])

            # [핵심 수정] out_transform을 사용해 잘라낸 이미지 기준의 좌표 변환
            topx, topy = rasterio.transform.xy(out_transform, r, c)

            results.append({
                "input_index": i,
                "input_x": pt.x,
                "input_y": pt.y,
                "polygon_index": int(nearest_crown_index),
                "tree_top_x": topx,
                "tree_top_y": topy,
                "tree_height": max_height,
                "centre_x": centroid.x,
                "centre_y": centroid.y
            })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Process complete. Saved {len(out)} rows to {output_csv}")
    print(out.head())

# 실행
if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    local_centroid_per_row(
        chm_tif="original/orange_trees.tif",
        crowns_shp="original/orange_trees.shp",
        input_csv="result/random_point.csv",
        output_csv="result/centre_points.csv",
        filter_size=7
    )