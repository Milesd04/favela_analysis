import osmnx as ox

# gdf = ox.geocode_to_gdf("Rio de Janeiro, Brazil")
# water = ox.geometries_from_place("Rio de Janeiro, Brazil", tags={"natural": "water"})

# gdf.to_file("rio_boundary.shp")
# water.to_file("rio_water.shp")


import osmnx as ox

# 1) Get the boundary of Rio de Janeiro state (by name)
# This can sometimes confuse city vs. state, so check results
gdf_boundary = ox.geocode_to_gdf("Rio de Janeiro (state), Brazil")

# 2) Grab coastline or water features in that boundary
# (But for big areas, it may time out)
# coastline = ox.geometries_from_place(
#     "Brazil",
#     tags={"natural": "coastline"}
# )

gdf_boundary.to_file("rio-shp")
# coastline.to_file("coastline.shp")

