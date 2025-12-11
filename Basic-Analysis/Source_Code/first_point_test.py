import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.ndimage import maximum_filter
from shapely.geometry import Point
from rasterio.mask import mask

input_img = r"..\Result\Orange_trees.tif"
input_random_csv = r"..\Result\random_point.csv"
input_ground_csv = r"..\Result\ground_truth.csv"
input_shapefile = r"..\Result\orange_trees.shp"
output_path = r"..\Result"

input_csv = pd.read_csv(input_ground_csv)
input_random = pd.read_csv(input_random_csv)

#Checking with the first coordinate
print(input_csv.iloc[0,:])
#First x from input_csv
first_csv = input_csv.iloc[0,0]
close_no_x = 0.0
close_no_y = 0.0
min_diff_x = float("inf")
min_diff_y = float("inf")
for index, row in input_random.iterrows():
    current_x = row.iloc[0]
    current_y = row.iloc[1]

    diff_x = abs(current_x-input_csv.iloc[0,0])
    diff_y = abs(current_y-input_csv.iloc[0,1])
    print(diff_x, " " , diff_y)

    if diff_x < min_diff_x and diff_y < min_diff_y:
        min_diff_x = diff_x
        min_diff_y = diff_y
        close_no_x = current_x
        close_no_y = current_y
        print("existing: ",min_diff_y,  " ", min_diff_x)
    

print("Closes Number : ", close_no_x, close_no_y)
print(f"{first_csv} : first")