import rasterio
import numpy as np
import os

tif_directory = '../Turing Dataset/Orthomosaicos/p4_transparent_mosaic_group1.tif'
output_tif = 'p4_quadtree_result.tif'

with rasterio.open(tif_directory) as src:
    data = src.read(1)
    profile = src.profile

threshold = 10

def quadtree_simplify(arr, x=0, y=0, size=None, threshold=10, out=None):
    if size is None:
        size = arr.shape[0]
    if out is None:
        out = np.zeros_like(arr)
        
    region = arr[y:y+size, x:x+size]
    
    if region.size == 0:
        return out
    
    std = np.std(region)
    
    if std < threshold or size <= 2:
        mean_val = np.mean(region)
        out[y:y+size, x:x+size] = mean_val
        return out
    
    half = size // 2
    quadtree_simplify(arr, x, y, half, threshold, out)
    quadtree_simplify(arr, x+half, y, half, threshold, out)
    quadtree_simplify(arr, x, y+half, half, threshold, out)
    quadtree_simplify(arr, x+half, y+half, half, threshold, out)
    
    return out

simplified = quadtree_simplify(data, threshold=threshold)

profile.update(dtype=rasterio.float32)

with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(simplified.astype(np.float32), 1)

print(f"✅ Quadtree 적용 완료!\n저장 위치: {output_tif}")
