import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

df = '../Dataset/Censo Forestal.csv'
df_five = '../Dataset/five_random_per_species.csv'
tif_repo = '../../../p4_transparent_mosaic_group1.tif'

save_tif_included = '../Dataset/features_tif_included.csv'
save_five_included = '../Dataset/five_features_tif_included.csv'

df = pd.read_csv(df)
df_five = pd.read_csv(df_five)
raster = rasterio.open(tif_repo)

bounds = raster.bounds

def feature_included_tif(input):
    print("TIF bounds:")
    print(f"left: {bounds.left}, right: {bounds.right}")
    print(f"bottom: {bounds.bottom}, top: {bounds.top}")

    # if feature exists based on the COORDENADA_ESTE, and COORDENADA_NORTE in TIF uav image file.
    mask = (
        (input['COORDENADA_ESTE']>=bounds.left)&
        (input['COORDENADA_ESTE']<=bounds.right)&
        (input['COORDENADA_NORTE']>=bounds.bottom)&
        (input['COORDENADA_NORTE']<=bounds.top)
    )

    df_in_tif = input[mask].copy()

    print(f"Total number of features : ", len(input))
    print(f"Total number of features in the TIF : ", len(df_in_tif))
    
    print(f"Saved record : {save_tif_included}")
    return df_in_tif.to_csv(save_tif_included, index=False)

def five_included_tif(input_five):
    print("TIF bounds:")
    print(f"left: {bounds.left}, right: {bounds.right}")
    print(f"bottom: {bounds.bottom}, top: {bounds.top}")

    # if feature exists based on the COORDENADA_ESTE, and COORDENADA_NORTE in TIF uav image file.
    mask = (
        (input_five['COORDENADA_ESTE']>=bounds.left)&
        (input_five['COORDENADA_ESTE']<=bounds.right)&
        (input_five['COORDENADA_NORTE']>=bounds.bottom)&
        (input_five['COORDENADA_NORTE']<=bounds.top)
    )

    df_in_tif = input_five[mask].copy()

    print(f"Total number of features : ", len(input_five))
    print(f"Total number of features in the TIF : ", len(df_in_tif))
    
    print(f"Saved record : {save_tif_included}")
    # return df_in_tif.to_csv(save_tif_included, index=False)

feature_included_tif(df)
five_included_tif(df_five)