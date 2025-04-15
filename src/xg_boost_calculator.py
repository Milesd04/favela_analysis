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
    Run an XGBoost classification with proper data handling and return comprehensive results.
    """
    # Set up metrics and data
    if metrics is None:
        metrics = [
            'area', 'perimeter', 'lal', 'neighbour_dist',
            'mean_interbuilding_distance', 'adjacency', 'corners', 'shape_idx',
            'facade_ratio', 'compact_weighted_axis',
            'squareness', 'square_compact', 'rectangularity', 'rect_idx',
            'perimeter_wall', 'num_neighbors', 'elongation', 'fractal',
            'cwa', 'CAR', 'tes_area', 'shared_walls', 'convexity', 'cell_alignment',
            'avg_weighted_dist'
        ]

    X = ground_truth_gdf[metrics]
    y = ground_truth_gdf['label']

    # Encode labels with explicit order
    label_encoder = LabelEncoder()
    label_encoder.fit(["not-favela", "favela"]) # 0=not-favela, 1=favela
    y_encoded = label_encoder.transform(y)

    # Split first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )

    # Impute missing values using training data statistics
    imputer = SimpleImputer(strategy='median') # look into
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_full_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns) # For final predictions

    # Train model
    xgb_model = XGBClassifier(n_estimators=n_estimators,
                            random_state=random_state,
                            eval_metric='logloss')
    xgb_model.fit(X_train_imputed, y_train)

    # Evaluate
    y_pred = xgb_model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred) * 100
    class_report = classification_report(y_test, y_pred,
                                       target_names=label_encoder.classes_,
                                       output_dict=True)

    # Generate predictions for full dataset
    ground_truth_gdf['predicted_label'] = label_encoder.inverse_transform(
        xgb_model.predict(X_full_imputed)
    )

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': metrics,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'model': xgb_model,
        'imputer': imputer,
        'label_encoder': label_encoder,
        'X_train_imputed': X_train_imputed,
        'X_test_imputed': X_test_imputed,
        'X_full_imputed': X_full_imputed,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'classification_report': class_report,
        'feature_importances': feature_importances,
        'updated_gdf': ground_truth_gdf,
        'metrics': metrics
    }

def save_xgboost_results(results, output_dir, site_name):
    """
    Save XGBoost results with complete reproducibility package.

    Output:
        [site_name]_xgb_model.json: Trained model (XGBoost native format)
        [site_name]_imputer.pkl: Fitted SimpleImputer for preprocessing new data
        [site_name]_label_encoder.pkl: Fitted LabelEncoder for label handling
        [site_name]_X_train_imputed.pkl: Imputed training features
        [site_name]_X_test_imputed.pkl: Imputed test features
        [site_name]_X_full_imputed.pkl: Imputed features for entire dataset
        [site_name]_y_*.npy: y arrays (train/test/pred)
        [site_name]_accuracy.json: Test accuracy
        [site_name]_classification_report.csv: Detailed classification metrics
        [site_name]_feature_importances.csv: Feature importance rankings
        [site_name]_updated_buildings.shp: Predictions with geometry (GeoPackage)
        [site_name]_metrics.json: List of metrics used for training
    """
    os.makedirs(output_dir, exist_ok=True)
    base_path = lambda suffix: os.path.join(output_dir, f"{site_name}_{suffix}")

    # Save core model components
    results['model'].save_model(base_path("xgb_model.json"))

    with open(base_path("imputer.pkl"), 'wb') as f:
        pickle.dump(results['imputer'], f)

    with open(base_path("label_encoder.pkl"), 'wb') as f:
        pickle.dump(results['label_encoder'], f)

    # Save processed datasets
    results['X_full_imputed'].to_pickle(base_path("X_full_imputed.pkl"))
    results['X_train_imputed'].to_pickle(base_path("X_train_imputed.pkl"))
    results['X_test_imputed'].to_pickle(base_path("X_test_imputed.pkl"))

    # Save target arrays
    for arr_type in ['y_train', 'y_test', 'y_pred']:
        np.save(base_path(f"{arr_type}.npy"), results[arr_type])

    # Save evaluation metrics
    with open(base_path("accuracy.json"), 'w') as f:
        json.dump({'accuracy': results['accuracy']}, f)

    pd.DataFrame(results['classification_report']).transpose().to_csv(
        base_path("classification_report.csv"), index=True
    )

    results['feature_importances'].to_csv(
        base_path("feature_importances.csv"), index=False
    )

    # Save metadata
    with open(base_path("metrics.json"), 'w') as f:
        json.dump({'metrics': results['metrics']}, f)

    # # Save spatial predictions
    # results['updated_gdf'].to_file(
    #     base_path("updated_buildings.gpkg"),
    #     driver='GPKG',
    #     layer='buildings',
    #     encoding='utf-8'
    # )

# TODO: ADD CODE TO AUTOMATICALLY GENERATE CONFUSION MATRIX BASED ON XGBOOST RESULTS

if __name__ == "__main__":
    # Example usage (for testing):
    # import geopandas as gpd
    # gdf = gpd.read_file("path/to/shapefile.shp")
    # results = run_xgboost(gdf)
    # save_xgboost_results(results, r"C:\Users\miles\favela_analysis\output\test_folder")


    # example to load
    # def load_xgboost_results(output_dir, site_name):
    # """Reload saved model package"""
    # base = lambda s: os.path.join(output_dir, f"{site_name}_{s}")

    # return {
    #     'model': XGBClassifier().load_model(base("xgb_model.json")),
    #     'imputer': pickle.load(open(base("imputer.pkl"), 'rb')),
    #     'label_encoder': pickle.load(open(base("label_encoder.pkl"), 'rb')),
    #     'data': {
    #         'X_full': pd.read_pickle(base("X_full_imputed.pkl")),
    #         'X_train': pd.read_pickle(base("X_train_imputed.pkl")),
    #         'y_train': np.load(base("y_train.npy")),
    #     },
    #     'metrics': json.load(open(base("metrics.json")))['metrics']
    # }
    pass
