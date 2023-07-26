
"""
Pronvinces NL
=============

This is a small vector dataset containing polygons of the provinces of the
Netherlands, including water, presented as geopandas GeoDataFrame.
"""

import xugrid

gdf = xugrid.data.provinces_nl()
gdf.plot()
