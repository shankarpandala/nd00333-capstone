from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset, Run
dataset_name = 'bmw_cars'
run = Run.get_context()
ws = run.experiment.workspace
# Get a dataset by name
bmw_cars = Dataset.get_by_name(workspace=ws, name=dataset_name)

# Load a TabularDataset into pandas DataFrame
df = bmw_cars.to_pandas_dataframe()
y = df.price
X = df.drop(['price'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_features', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--min_samples_split', type=int)

    args = parser.parse_args()
    # b. get the names of the numeric anc categorical columns
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x_train.select_dtypes(include=['object', 'category', 'period[M]']).columns
    # create a transformer for the categorical values
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot', OneHotEncoder())])

    # create a transformed for the numerical values
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=args.n_estimators, max_features=args.max_features, max_depth=args.max_depth, min_samples_split=args.min_samples_split))])

    model = clf.fit(x_train, y_train)
    metric = model.score(x_test, y_test)
    run.log("normalized_root_mean_squared_error", np.float(metric))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/hd-model.joblib')

if __name__ == '__main__':
    main()