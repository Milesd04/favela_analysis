import geopandas as gpd

def create_complete_buildings(clean_buildings, centroids_gdf, tessellation):
    """
    Create a complete buildings GeoDataFrame by merging buildings with centroids and tessellation data.

    Parameters:
        clean_buildings (GeoDataFrame)
        centroids_gdf (GeoDataFrame)
        tessellation (GeoDataFrame)

    Returns:
        GeoDataFrame: A merged GeoDataFrame containing all building, centroid, and tessellation data
    """
    # Merge buildings and centroids_gdf
    buildings_with_centroids = clean_buildings.merge(
        centroids_gdf,
        on='uID',
        how='left',
        suffixes=('_building', '_centroid')
    )

    # Merge with tessellation
    complete_buildings = buildings_with_centroids.merge(
        tessellation,
        on='uID',
        how='left',
        suffixes=('', '_tessellation')
    )

    # Ensure geometry column is preserved
    complete_buildings = gpd.GeoDataFrame(
        complete_buildings,
        geometry='geometry_building',
        crs=clean_buildings.crs
    )

    return complete_buildings

def create_joined_gdf(clean_buildings, centroids_gdf, tessellation):
    """
    Create a joined GeoDataFrame using the join method.

    Parameters:
        clean_buildings (GeoDataFrame)
        centroids_gdf (GeoDataFrame)
        tessellation (GeoDataFrame)

    Returns:
        GeoDataFrame: A joined GeoDataFrame containing all building, centroid, and tessellation data
    """
    # Join buildings with centroids
    joined_gdf = clean_buildings.join(
        centroids_gdf,
        how='left',
        lsuffix='_building',
        rsuffix='_centroid'
    )

    # Join with tessellation
    joined_gdf = joined_gdf.join(
        tessellation,
        how='left',
        lsuffix='',
        rsuffix='_tessellation'
    )

    return joined_gdf