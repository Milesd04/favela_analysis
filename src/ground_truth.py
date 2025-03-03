import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def classify_and_analyze_favelas(clean_buildings, type_column='tipo', plot=True):
    """
    Classify buildings as favela/not-favela and provide analysis.

    Parameters:
        buildings_gdf (GeoDataFrame): Buildings data
        type_column (str): Name of column containing building type codes
        plot (bool): Whether to create visualization

    Returns:
        tuple:
            GeoDataFrame: Buildings with added classification
            dict: Analysis results
    """
    # Copy to avoid modifying original
    buildings_copy = clean_buildings.copy()

    buildings_copy['label'] = np.where(
        buildings_copy[type_column].notna(), # Check if there's any value
        'favela',
        'not-favela'
    )

    stats = {
        'total_buildings': len(buildings_copy),
        'favela_count': (buildings_copy['label'] == 'favela').sum(),
        'non_favela_count': (buildings_copy['label'] == 'not-favela').sum(),
        'unique_types': buildings_copy[type_column].unique().tolist(),
        'type_counts': buildings_copy[type_column].value_counts().to_dict()
    }

    stats['favela_percentage'] = (stats['favela_count'] / stats['total_buildings']) * 100

    print("\n=== Favela Classification Analysis ===")
    print(f"Total buildings: {stats['total_buildings']}")
    print(f"Favela buildings: {stats['favela_count']} ({stats['favela_percentage']:.1f}%)")
    print(f"Non-favela buildings: {stats['non_favela_count']}")

    print("\nBuilding type distribution:")
    for type_code, count in stats['type_counts'].items():
        print(f"{type_code}: {count} buildings")

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot 1: Spatial distribution
        buildings_copy.plot(ax=ax1, column='label', categorical=True,
                          legend=True, cmap='Set3',
                          legend_kwds={'title': 'Classification'})
        ax1.set_title('Spatial Distribution of Favelas')
        ax1.set_axis_off()

        # Plot 2: Classification counts
        pd.Series([stats['favela_count'], stats['non_favela_count']],
                  index=['favela', 'not-favela']).plot(kind='bar', ax=ax2,
                  color=['#FF6347', '#4682B4'])
        ax2.set_title('Favela vs Non-Favela Count')
        ax2.set_xlabel('Building Type')
        ax2.set_ylabel('Count')
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

    return buildings_copy, stats
