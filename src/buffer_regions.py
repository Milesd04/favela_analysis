import geopandas as gpd
from shapely.geometry import box, Point

def process_building_shapefile(input_shp, output_shp, center, initial_side_length=750,
                               k=0.5, max_iters=100, target_ratio=(0.25, 0.35), count_column='nome_count',
                               ):
    """
    Process a building shapefile to label buildings and extract those within a region
    that meets a target favela ratio.

    Parameters:
        input_shp (str): Path to input building shapefile.
        output_shp (str): Path where output shapefile will be saved.
        center (Point): Fixed center point for the study region.
        initial_side_length (float): Starting side length for the square region in meters.
        k (float): Tuning parameter for dynamic scaling.
        max_iters (int): Max number of iterations for adjusting the square.
        target_ratio (tuple): Desired ratio range (min, max) for favela buildings.
        count_column (str): Column name used to determine building type.

    Returns:
        GeoDataFrame: The subset of buildings that meet the criteria.
    """
    gdf = load_buildings_within_radius(input_shp, center_point, radius=1500) # how big should this be?
    gdf = label_buildings(gdf, count_column=count_column) # favela or not-favela

    best_square = adjust_square(gdf, center, initial_side_length, k, max_iters, target_ratio)

    buildings_subset = extract_buildings_within_square(gdf, best_square)
    print(f"Number of buildings in best region: {len(buildings_subset)}")

    # Save output shapefile
    buildings_subset.to_file(output_shp)
    print(f"Saved filtered buildings to: {output_shp}")

    return buildings_subset

def label_buildings(gdf, count_column='nome_count'):
    """
    Label buildings based on the nome count column.

    Buildings with a value of 1 or 2 are labeled as 'favela',
    all others (including null values) are labeled as 'not-favela'.

    Parameters:
        gdf (GeoDataFrame): The building data.
        count_column (str): Column name for building count.

    Returns:
        GeoDataFrame: A copy of gdf with an added 'label' column.
    """
    gdf = gdf.copy()
    gdf['label'] = gdf[count_column].apply(lambda x: 'favela' if x in [1, 2] else 'not-favela')
    return gdf

def compute_square(center, half_side):
    """
    Create a square polygon centered at a given point.

    Parameters:
        center (Point): Center point of the square.
        half_side (float): Half of the square's side length (in the same units as center).

    Returns:
        Polygon: The resulting square polygon.
    """
    return box(center.x - half_side, center.y - half_side,
               center.x + half_side, center.y + half_side)

def adjust_square(gdf, center, initial_side_length, k=0.1, max_iters=250, target_ratio=(0.25, 0.30)):
    """
    Iteratively adjust a square region to achieve a target ratio of favela buildings.

    Parameters:
        gdf (GeoDataFrame): Building data with a 'label' column.
        center (Point): Fixed center for the square.
        initial_side_length (float): Starting square side length.
        k (float): Tuning parameter for dynamic scaling.
        max_iters (int): Maximum iterations to adjust the square.
        target_ratio (tuple): Target ratio range (min, max) for favela buildings.

    Returns:
        Polygon: The square polygon meeting (or approximating) the target ratio.
    """
    current_half_side = initial_side_length / 2
    best_square = None

    for i in range(max_iters):
        square = compute_square(center, current_half_side)
        buildings_in_square = gdf[gdf.geometry.within(square)]
        total = len(buildings_in_square)

        if total == 0:
            print(f"Iteration {i+1}: No buildings found; aborting.")
            break

        favela_count = len(buildings_in_square[buildings_in_square['label'] == 'favela'])
        current_ratio = favela_count / total
        print(f"Iteration {i+1}: Total = {total}, Favela = {favela_count}, Ratio = {current_ratio:.2f}")

        # best square reached
        if target_ratio[0] <= current_ratio <= target_ratio[1]:
            print("Desired ratio reached.")
            best_square = square
            return best_square

        # dynamic scaling based on how far the current ratio is from target range
        if current_ratio < target_ratio[0]:
            error = target_ratio[0] - current_ratio
            factor = 1 - k * error
        else:
            error = current_ratio - target_ratio[1]
            factor = 1 + k * error

        current_half_side *= factor
        best_square = square

    print("Desired ratio approximated")
    return best_square

def extract_buildings_within_square(gdf, square):
    """
    Extract buildings from GeoDataFrame that lie within the given square.

    Parameters:
        gdf (GeoDataFrame): The building data.
        square (Polygon): Square polygon used as a mask.

    Returns:
        GeoDataFrame: Buildings that satisfy spatial predicate.
    """
    return gdf[gdf.geometry.intersects(square)]

def load_buildings_within_radius(input_shp, center, radius=1500):
    """
    Load features from a shapefile that intersect a bounding box around a given center point.

    Parameters:
        input_shp (str): Path to the input shapefile.
        center (shapely.geometry.Point): Center point.
        radius (float): Distance in same units as center's CRS (e.g., meters) to use as half the side of the bounding box.

    Returns:
        GeoDataFrame: Features that intersect the bounding box.
    """
    # Calculate the bounding box based on the center and radius.
    minx, miny = center.x - radius, center.y - radius
    maxx, maxy = center.x + radius, center.y + radius
    bbox = (minx, miny, maxx, maxy)

    # Load only features within the bounding box.
    gdf = gpd.read_file(input_shp, bbox=bbox)
    print(gdf.columns)
    return gdf

# Example usage
if __name__ == "__main__":
    #explain use
    # input_shapefile = r"C:\Users\miles\OneDrive\Desktop\school\UROP\Data\Test Data\nova_cidade.shp"
    # output_shapefile = r"C:\Users\miles\favela_analysis\output"
    # center_point = Point(644547.4, 7466496.6)  # Example center (in EPSG:31983)
    # initial_length = 750  # Starting side length in meters

    input_shapefile = r"C:\Users\miles\OneDrive\Desktop\school\UROP\Data\Favelas_joined.shp"
    # output_shapefile = r"C:\Users\miles\favela_analysis\site_shps\jacarezinho.shp"
    # center_point = Point(678527.25, 7467819.51)  # Example center (in EPSG:31983) - Jacarezinho


    output_shapefile = r"C:\Users\miles\favela_analysis\site_shps\morro_da_guaiba.shp"
    center_point = Point(674751.62, 7473736.83)

    filtered_buildings = process_building_shapefile(
        input_shp=input_shapefile,
        output_shp=output_shapefile,
        center=center_point,
        initial_side_length=200,
        k=0.5,
        max_iters=30,
        target_ratio=(0.25, 0.35),
        count_column='nome_count',
    )
