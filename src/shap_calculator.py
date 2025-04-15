import os
import pandas as pd
import shap
import numpy as np

def calculate_shap_results(xgb_model, X_imputed, metrics, ground_truth_gdf):
    """
    Compute SHAP values for a trained XGBoost model and return a dictionary of results.

    Parameters:
        xgb_model (XGBClassifier): Trained XGBoost model.
        X_imputed (pd.DataFrame): DataFrame of imputed feature values used for training/prediction.
        metrics (list of str): List of feature names used for training.
        ground_truth_gdf (GeoDataFrame): Original GeoDataFrame containing building data.

    Returns:
        dict:
            'explainer' (Explainer): The SHAP explainer object.
            'shap_values' (NumPy array): The raw SHAP values as a NumPy array.
            'shap_df' (DataFrame): SHAP values with columns named after the metrics.
            'updated_gdf' (GeoDataFrame): Updated with new columns for each feature's SHAP values.
    """
    # Initialize explainer with correct parameters
    explainer = shap.TreeExplainer(
        xgb_model,
        model_output="raw" # XGBoost's raw margin (log-odds) output
    )

    # Get SHAP values for class 1 (favela)
    shap_values = explainer.shap_values(X_imputed)
    # reverse for visualization
    shap_values = -shap_values

    # Handle binary classification output format

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1] # [0]=not-favela, [1]=favela chnag ing this value doesn't do anything lol

    # Create DataFrame and update GeoDataFrame
    shap_df = pd.DataFrame(shap_values, columns=metrics)
    shap_df.index = ground_truth_gdf.index

    updated_gdf = ground_truth_gdf.copy()
    for feature in metrics:
        updated_gdf[f'shap_{feature}'] = shap_df[feature]

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'shap_df': shap_df,
        'updated_gdf': updated_gdf
    }

def save_shap_results(shap_results, output_dir, site_name):
    """
    Save the SHAP results to the specified output directory.

    Parameters:
        shap_results (dict):
            Dictionary containing 'shap_df', 'shap_values', 'updated_gdf', and 'explainer'.
        output_dir (str):
            The path to the directory where all output files should be saved.
        site_name (str):
            A short identifier used in the generated filenames.

    Output:
        [site_name]_shap_values_only.pkl:
            The SHAP values (from shap_df) serialized as a pandas DataFrame.
        [site_name]_shap_values_full.npy:
            The raw SHAP values as a NumPy array.
        [site_name]_updated_buildings_with_shap.gpkg:
            The updated GeoDataFrame (with SHAP columns) saved as a GeoPackage.
        [site_name]_explainer.shap:
            (Commented out by default) The SHAP explainer saved via shap.save().
    """

    shap_pkl_path = os.path.join(output_dir, f"{site_name}_shap_values_only.pkl")
    shap_results['shap_df'].to_pickle(shap_pkl_path)

    shap_values_npy_path = os.path.join(output_dir, f"{site_name}_shap_values_full.npy")
    shap_values = shap_results['shap_values']
    np.save(shap_values_npy_path, shap_values)

    updated_gdf = shap_results['updated_gdf'].copy()
    updated_gdf = updated_gdf.set_geometry("geometry")

    # drop extra geometry columns if they exist
    for extra_geom in ['centroid', 'geometry_tessellation']:
        if extra_geom in updated_gdf.columns:
            updated_gdf = updated_gdf.drop(columns=[extra_geom])

    geopackage_path = os.path.join(output_dir, f"{site_name}_updated_buildings_with_shap.gpkg")
    updated_gdf.to_file(geopackage_path, driver='GPKG')

    # Save the explainer using SHAP's built-in save function
    # SHAP website suggests using: save(out_file[, model_saver, masker_saver])
    # explainer_path = os.path.join(site_subfolder, site_name + '_explainer.shap')
    # with open(explainer_path, 'wb') as f:
    #     shap_results['explainer'].save(f, model_saver=dummy)

def dummy(output_dir, model):
    pass

if __name__ == "__main__":
    # Example usage:
    # from xg_boost_calculator import run_xgboost
    # import geopandas as gpd
    # gdf = gpd.read_file("path/to/shapefile.shp")
    # results = run_xgboost(gdf)
    # from shap_calculator import calculate_shap_results, save_shap_results
    # shap_results = calculate_shap_results(results['model'], results['X_imputed'], results['metrics'], results['updated_gdf'])
    # save_shap_results(shap_results, r"C:\Users\miles\favela_analysis\output\test_folder")
    pass


