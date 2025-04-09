import os
import pandas as pd
import shap
import numpy as np

### TO DO: NEED TO MAKE IT SO THAT THE GDF SAVES THE ENTIRE NAME OF THE GIVEN FIELD

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
    explainer = shap.Explainer(xgb_model, X_imputed)
    shap_values = explainer(X_imputed, check_additivity=False) #check_additivity = False for now... might be worth investigating this. the reported error difference was negligible <0.1
    shap_df = pd.DataFrame(shap_values.values, columns=metrics)
    shap_df.index = ground_truth_gdf.index
    updated_gdf = ground_truth_gdf.copy()
    for feature in metrics:
        updated_gdf[f'shap_{feature}'] = shap_df[feature]

    result = {
        'explainer': explainer,
        'shap_values': shap_values, # i may have broke this by changing it from .values() but we'll see
        'shap_df': shap_df,
        'updated_gdf': updated_gdf
    }

    return result


def save_shap_results(shap_results, output_dir, site_name):
    """
    Save the SHAP results to the specified output directory.

    The following files will be created in output_dir (inside a subfolder named site_name):
        shap_values.pkl: the SHAP values as a pkl file (from shap_df)
        shap_values.pkl: the raw SHAP values
        updated_buildings_with_shap.shp: the GeoDataFrame with SHAP columns (saved as a shapefile)
        explainer.shap: the SHAP explainer saved to a file stream using shap.save()

    Parameters:
        shap_results (dict)
        output_dir (str)
        site_name (str)
    """
    # Create a subfolder named after site_name
    site_subfolder = os.path.join(output_dir, site_name)
    os.makedirs(site_subfolder, exist_ok=True)

    # Save the SHAP DataFrame as pkl.
    shap_pkl_path = os.path.join(site_subfolder, site_name + '_shap_values_only.pkl')
    shap_results['shap_df'].to_pickle(shap_pkl_path)

    # Save the raw SHAP values as npy, as it is a numpy.ndarray
    shap_values_npy_path = os.path.join(site_subfolder, site_name + '_shap_values_full.npy')
    shap_values = shap_results['shap_values'].values
    np.save(shap_values_npy_path, shap_values)

    # Prepare the updated GeoDataFrame for saving
    updated_gdf = shap_results['updated_gdf'].copy()
    updated_gdf = updated_gdf.set_geometry("geometry")

    # Drop extra geometry columns if they exist
    for extra_geom in ['centroid', 'geometry_tessellation']:
        if extra_geom in updated_gdf.columns:
            updated_gdf = updated_gdf.drop(columns=[extra_geom])

    # Save the cleaned GeoDataFrame as a shapefile.
    shapefile_path = os.path.join(site_subfolder, site_name + '_updated_buildings_with_shap.shp')
    updated_gdf.to_file(shapefile_path)

    # Save the explainer using SHAP's built-in save function
    # SHAP website suggests using: save(out_file[, model_saver, masker_saver])
    explainer_path = os.path.join(site_subfolder, site_name + '_explainer.shap')
    with open(explainer_path, 'wb') as f:
        shap_results['explainer'].save(f, model_saver=dummy)

    print("SHAP results successfully saved to:", site_subfolder)

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


