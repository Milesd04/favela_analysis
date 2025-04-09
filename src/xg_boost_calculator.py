import os
import json
import os, pickle, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def run_xgboost(ground_truth_gdf, metrics=None, test_size=0.2, random_state=432, n_estimators=100):
    """
    Run an XGBoost classification on a GeoDataFrame using specified features and return the model, evaluation metrics,
    feature importances, and the updated GeoDataFrame with predictions.

    Parameters:
        ground_truth_gdf (GeoDataFrame): Input GeoDataFrame containing building data with a 'label' column.
        metrics (list of str, optional): List of feature columns to use for training. If None, a default list is used.
        test_size (float): Proportion of the data to set aside for testing.
        random_state (int): Random seed for reproducibility.
        n_estimators (int): Number of trees (estimators) for the XGBoost model.

    Returns:
        dict: A dictionary containing:
            'model': The trained XGBoost model.
            'X_imputed': The imputed features DataFrame.
            'X_train': Training set features.
            'X': The original feature DataFrame.
            'accuracy' (float)
            'classification_report' (str): Results of classification
            'feature_importances' (DataFrame)
            'updated_gdf' (GeoDataFrame): Updated with new column 'predicted_label'.
            'metrics' (lst): List of metrics used for training.
    """
    if metrics is None:
        metrics = [
            'area', 'perimeter', 'lal', 'neighbour_dist',
            'mean_interbuilding_distance', 'adjacency', 'corners', 'shape_idx',
            'facade_ratio', 'compact_weighted_axis',
            'squareness', 'square_compact', 'rectangularity', 'rect_idx',
            'perimeter_wall', 'num_neighbors', 'elongation', 'fractal',
            'cwa', 'CAR', 'tes_area', 'shared_walls', "convexity", "cell_alignment",
        ]

    X = ground_truth_gdf[metrics]
    y = ground_truth_gdf['label']

    label_encoder = LabelEncoder()
    label_encoder.fit(['not-favela', 'favela'])
    y_encoded = label_encoder.transform(y)
    y_encoded = 1 - y_encoded  # so that not-favela = 0, favela = 1 (hacky i should figure out how this works)

    print("Target value counts:")
    print(pd.Series(y_encoded).value_counts())

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=test_size, random_state=random_state)

    xgb_model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    ground_truth_gdf['predicted_label'] = label_encoder.inverse_transform(xgb_model.predict(X_imputed))

    accuracy = accuracy_score(y_test, y_pred) * 100
    class_report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(class_report_dict)

    feature_importances = pd.DataFrame({
        'Feature': metrics,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importances:")
    print(feature_importances)

    results = {
        'model': xgb_model,
        'X_imputed': X_imputed,
        'X_train': X_train,
        'X': X,
        'X_test': X_test,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_train': y_train,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'classification_report': class_report_dict,
        'feature_importances': feature_importances,
        'updated_gdf': ground_truth_gdf,
        'metrics': metrics
    }

    return results

def save_xgboost_results(results, output_dir, site_name):
    """
    Save the XGBoost results to the specified output directory.

    The following files are created in output_dir:
        xgb_model.json: the trained model
        X_imputed.pkl: imputed feature values
        X_train.pkl: training features
        X.pkl: original features
        y_pred.pkl:
        y_test.pkl:
        y_train.pkl:
        accuracy.json: a JSON file with the accuracy value
        classification_report.txt: the classification report as text
        feature_importances.csv: feature importance values
        updated_buildings.shp: the updated GeoDataFrame with predictions saved as a shapefile
        metrics.json: the list of metrics used (in JSON format)
    """


    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model.
    model_filepath = os.path.join(output_dir, site_name + '_xgb_model.json')
    results['model'].save_model(model_filepath)

    # Save label encoder
    label_encoder_filepath = os.path.join(output_dir, site_name + '_label_encoder_classes.npy')
    np.save(label_encoder_filepath, results['label_encoder'].classes_)

    #CAN LOAD WITH:
    # from xgboost import XGBClassifier
    # model = XGBClassifier()
    # model.load_model(model_filepath)

    # Save DataFrames.
    results['X_imputed'].to_pickle(os.path.join(output_dir, site_name + '_X_imputed.pkl'))
    results['X_train'].to_pickle(os.path.join(output_dir, site_name + '_X_train.pkl'))
    results['X'].to_pickle(os.path.join(output_dir, site_name + '_X.pkl'))
    results['X_test'].to_pickle(os.path.join(output_dir, site_name + '_X_test.pkl'))

    y_test_path = os.path.join(output_dir, site_name + '_y_test.npy')
    y_pred_path = os.path.join(output_dir, site_name + '_y_pred.npy')
    y_train_path = os.path.join(output_dir, site_name + '_y_train.npy')

    np.save(y_test_path, results['y_test'])
    np.save(y_pred_path, results['y_pred'])
    np.save(y_train_path, results['y_train'])

    # Save accuracy and classification report.
    with open(os.path.join(output_dir, site_name + '_accuracy.json'), 'w') as f:
        json.dump({'accuracy': results['accuracy']}, f)

    # Save the classification report as a CSV file
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_df.to_csv(os.path.join(output_dir, site_name + '_classification_report.csv'), index=True)

    # Save feature importances.
    results['feature_importances'].to_csv(os.path.join(output_dir, site_name + '_feature_importances.csv'))

    # Save the metrics list.
    with open(os.path.join(output_dir, site_name + '_metrics.json'), 'w') as f:
        json.dump({'metrics': results['metrics']}, f)

# TODO: ADD CODE TO AUTOMATICALLY GENERATE CONFUSION MATRIX BASED ON XGBOOST RESULTS

if __name__ == "__main__":
    # Example usage (for testing):
    # import geopandas as gpd
    # gdf = gpd.read_file("path/to/shapefile.shp")
    # results = run_xgboost(gdf)
    # save_xgboost_results(results, r"C:\Users\miles\favela_analysis\output\test_folder")
    pass
