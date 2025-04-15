import pandas as pd
import geopandas as gpd
import momepy as mm
from shapely.geometry import Polygon
import warnings

def perform_perfect_tessellation(buildings_path, min_area=1.0, buffer_dist=100, max_iterations=10):
    """
    Performs morphological tessellation ensuring perfect 1:1 matching between buildings and tessellation cells.
    Uses a single-pass approach to handle problematic geometries proactively.

    Parameters:
        buildings_path (str): Path to the buildings shapefile
        min_area (float): Minimum area threshold for buildings
        buffer_dist (float): Distance for buffered limit
        max_iterations (int): Maximum number of iterations (kept for API compatibility) TODO get rid of this :skull:

    Returns:
        tuple: (clean_buildings, tessellation, excluded_buildings, report)
    """
    report = {
        'original_count': 0,
        'excluded_count': 0,
        'final_count': 0,
        # 'iterations_needed': 1,
        'reasons_for_exclusion': {},
        'tessellation_count': 0
    }

    # Read buildings
    buildings = gpd.read_file(buildings_path)
    report['original_count'] = len(buildings)

    # Create a copy for tracking excluded buildings
    excluded_buildings = gpd.GeoDataFrame(columns=buildings.columns + ['exclusion_reason'])

    def is_suitable_geometry(geom):
        if geom is None:
            return False, "null_geometry"
        if not geom.is_valid:
            return False, "invalid_geometry"
        if geom.is_empty:
            return False, "empty_geometry"
        if not isinstance(geom, Polygon):
            return False, "not_polygon"
        if geom.area < min_area:
            return False, "too_small"
        return True, "suitable"

    # Initial geometry cleaning
    clean_buildings = buildings.copy()
    exclusion_mask = pd.Series(False, index=buildings.index)
    reasons = []

    # First pass: remove problematic geometries
    for idx, geom in buildings.geometry.items():
        is_suitable, reason = is_suitable_geometry(geom)
        if not is_suitable:
            exclusion_mask.loc[idx] = True
            reasons.append(reason)
            report['reasons_for_exclusion'][reason] = report['reasons_for_exclusion'].get(reason, 0) + 1

    # Separate excluded buildings
    initially_excluded = buildings[exclusion_mask].copy()
    initially_excluded['exclusion_reason'] = reasons
    excluded_buildings = pd.concat([excluded_buildings, initially_excluded])

    # Keep only clean buildings
    clean_buildings = buildings[~exclusion_mask].copy()
    clean_buildings = clean_buildings.reset_index(drop=True)

    if len(clean_buildings) == 0:
        raise Exception("No clean buildings remaining after filtering")

    try:
        # Create buffered limit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            limit = mm.buffered_limit(clean_buildings, buffer_dist)

        clean_buildings['uID'] = mm.unique_id(clean_buildings)

        # Create initial tessellation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            tessellation = mm.morphological_tessellation(clean_buildings, clip=limit)

        # Identify buildings without corresponding tessellation cells
        tessellation.index = tessellation.index.astype(int)
        missing_uids = set(clean_buildings['uID']) - set(tessellation.index)

        if missing_uids:
            # For buildings without tessellation cells, check if they're problematic
            problem_buildings = clean_buildings[clean_buildings['uID'].isin(missing_uids)].copy()
            problem_buildings['exclusion_reason'] = 'tessellation_failure'
            excluded_buildings = pd.concat([excluded_buildings, problem_buildings])

            # Update clean buildings
            clean_buildings = clean_buildings[~clean_buildings['uID'].isin(missing_uids)].copy()
            clean_buildings = clean_buildings.reset_index(drop=True)

            # Update tessellation
            tessellation = tessellation[tessellation.index.isin(clean_buildings['uID'])].copy()
            tessellation = tessellation.reset_index(drop=True)

            report['reasons_for_exclusion']['tessellation_failure'] = \
                report['reasons_for_exclusion'].get('tessellation_failure', 0) + len(problem_buildings)

    except Exception as e:
        raise Exception(f"Error during tessellation: {str(e)}")

    # Update final report
    report['excluded_count'] = len(excluded_buildings)
    report['final_count'] = len(clean_buildings)
    report['tessellation_count'] = len(tessellation)

    # Final validation
    if len(clean_buildings) != len(tessellation):
        raise Exception("Failed to achieve perfect matching")

    return clean_buildings, tessellation, excluded_buildings, report

def print_perfect_tessellation_report(report):
    """
    Print a formatted report of the tessellation results.

    Parameters:
    report (dict): Dictionary containing tessellation statistics
    """
    print("\nTessellation Report:")
    print("-" * 50)
    print(f"Original building count: {report['original_count']}")
    print(f"Excluded building count: {report['excluded_count']}")
    print(f"Final building count: {report['final_count']}")
    print(f"Tessellation cell count: {report['tessellation_count']}")
    # print(f"Iterations needed: {report['iterations_needed']}")

    print("\nReasons for exclusion:")
    if report['reasons_for_exclusion']:
        for reason, count in report['reasons_for_exclusion'].items():
            print(f"  - {reason}: {count} buildings")
    else:
        print("No buildings excluded! :)")