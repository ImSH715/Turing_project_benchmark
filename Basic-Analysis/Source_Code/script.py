import shapefile
from shapely.geometry import Polygon
import csv
import random

sf = shapefile.Reader(
    r"Z:\ai4eo\Shared\2025_Turing_L\datasets\orange_trees\orange_trees.shp")

print(sf)
print(sf.bbox)

shapes = sf.shapes()
print(shapes[0].points)

poly = Polygon(shapes[0].points)
print(poly.centroid)

with open('ground_truth.csv', 'w', newline='') as csvfile:
    truthwriter = csv.writer(csvfile, delimiter=',',
                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
    truthwriter.writerow(['lat', 'long'])
    for shape in shapes:
        poly = Polygon(shape.points)
        truthwriter.writerow(
            poly.centroid.coords[0])

with open('random_point.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['lat', 'long'])
    for shape in shapes:
        poly = Polygon(shape.points)
        coords = poly.centroid.coords[0]
        writer.writerow(
            [coords[0] + random.uniform(-1.5, 1.5), coords[1] + random.uniform(-1.5, 1.5)])

print(sf.bbox[2]-sf.bbox[0], sf.bbox[3]-sf.bbox[1])
