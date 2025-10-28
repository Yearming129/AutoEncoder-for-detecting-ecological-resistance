# -*- coding: utf-8 -*-
# @Filename : 4.py

from osgeo import gdal,osr

# gcp_items = [
#     [76636.8490356412, 3387085.45109667, 0, 0],         # 左上，0行0列
#     [645076.849035641, 3387085.45109667, 18948, 0],     # 右上，0行x列
#     [76636.8490356412, 3006985.45109667, 0, 12670],       # 左下，y行0列
#     [645076.849035641, 3006985.45109667, 12670, 18948]    # 右下，y行x列
# ]
#
# gcp_list = []
# for item in gcp_items:
#     x, y, pixel, line = item
#     z = 0
#     gcp = gdal.GCP(x, y, z, pixel, line)
#     gcp_list.append(gcp)
#
# options = gdal.TranslateOptions(format='GTiff', outputSRS='EPSG:9122',GCPs=gcp_list)
# gdal.Translate('new2_image.tiff', 'new_image.tiff', options=options)

tif_path = r'C:\Users\MR\Desktop\AE\auto_encoder\1\new_image2020.tiff'
dataset = gdal.Open(tif_path)
pcs = osr.SpatialReference()
print(pcs)
pcs.ImportFromWkt(dataset.GetProjection())
gcs = pcs.CloneGeogCS()
print(gcs)
extend = dataset.GetGeoTransform()
print(extend)