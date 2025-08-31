import sys
import logging
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Core scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# --- 1. The Custom Transformer with All Preprocessing Logic ---
# This class handles all the complex steps while preserving column names.
class ComprehensivePreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that performs a sequence of operations:
    1. Imputes missing values.
    2. One-Hot Encodes categorical features.
    3. Selects features using a Chi-Square test against the target.
    4. Removes multicollinearity using an iterative VIF check.
    """
    def __init__(self, chi2_p_value_threshold=0.05, vif_threshold=5.0):
        self.chi2_p_value_threshold = chi2_p_value_threshold
        self.vif_threshold = vif_threshold
        
        # Learned attributes
        self.encoder_ = None
        self.imputer_num_ = None
        self.imputer_cat_ = None
        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.final_features_ = []

    def fit(self, X, y):
        logging.info("Fitting the ComprehensivePreprocessor.")
        X_processed = X.copy()

        # Step 0: Identify column types
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        # Step 1: Fit Imputers
        self.imputer_num_ = SimpleImputer(strategy='median')
        self.imputer_cat_ = SimpleImputer(strategy='most_frequent')
        X_processed[self.numerical_cols_] = self.imputer_num_.fit_transform(X_processed[self.numerical_cols_])
        X_processed[self.categorical_cols_] = self.imputer_cat_.fit_transform(X_processed[self.categorical_cols_])

        # Step 2: Fit Encoder
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        self.encoder_.fit(X_processed[self.categorical_cols_])
        
        # Transform data to apply subsequent tests
        encoded_cols = self.encoder_.get_feature_names_out(self.categorical_cols_)
        X_encoded_df = pd.DataFrame(self.encoder_.transform(X_processed[self.categorical_cols_]), columns=encoded_cols, index=X.index)
        X_processed = pd.concat([X_processed[self.numerical_cols_], X_encoded_df], axis=1)

        # Step 3: Chi-Square Feature Selection
        chi2_selected_features = []
        for feature in X_processed.columns:
            contingency_table = pd.crosstab(X_processed[feature], y)
            _, p_val, _, _ = chi2_contingency(contingency_table)
            if p_val <= self.chi2_p_value_threshold:
                chi2_selected_features.append(feature)
        X_chi2 = X_processed[chi2_selected_features]

        # Step 4: VIF for Multicollinearity (Iterative Removal)
        features_to_keep = list(X_chi2.columns)
        while True:
            if len(features_to_keep) < 2:
                break
            vif_data = pd.DataFrame()
            vif_data["feature"] = features_to_keep
            vif_data["VIF"] = [variance_inflation_factor(X_chi2[features_to_keep].values, i) for i in range(len(features_to_keep))]
            max_vif = vif_data['VIF'].max()
            if max_vif > self.vif_threshold:
                feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
                features_to_keep.remove(feature_to_drop)
            else:
                break

        self.final_features_ = features_to_keep
        logging.info(f"Preprocessor fitting complete. Final features selected: {self.final_features_}")
        return self

    def transform(self, X):
        logging.info("Transforming data using the fitted preprocessor.")
        X_processed = X.copy()
        
        # Apply imputers
        X_processed[self.numerical_cols_] = self.imputer_num_.transform(X_processed[self.numerical_cols_])
        X_processed[self.categorical_cols_] = self.imputer_cat_.transform(X_processed[self.categorical_cols_])

        # Apply encoder
        encoded_cols = self.encoder_.get_feature_names_out(self.categorical_cols_)
        X_encoded_df = pd.DataFrame(self.encoder_.transform(X_processed[self.categorical_cols_]), columns=encoded_cols, index=X.index)
        X_processed = pd.concat([X_processed[self.numerical_cols_], X_encoded_df], axis=1)

        # Return only the final selected features
        return X_processed[self.final_features_]


# --- 2. The Main Class that Builds and Returns the Pipeline ---
class DataTransformation:
    """
    This class is responsible for creating and returning the data transformation pipeline.
    """
    def __init__(self):
        # You could pass configuration here if needed
        pass

    def get_data_preprocessor_object(self) -> Pipeline:
        """
        This method builds the complete preprocessing pipeline and returns it.
        """
        logging.info("Creating data preprocessing pipeline object.")
        try:
            # The pipeline consists of just one step: our powerful custom preprocessor
            pipeline = Pipeline(
                steps=[
                    ('preprocessor', ComprehensivePreprocessor(
                        chi2_p_value_threshold=0.05, 
                        vif_threshold=5.0
                    ))
                ]
            )
            logging.info("Pipeline created successfully.")
            return pipeline

        except Exception as e:
            # Assuming you have a custom exception class
            logging.error(f"Error creating pipeline: {e}")
            raise e # Or your custom exception


# --- 3. Example of How to Use the Final Pipeline ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy data with various issues to test the pipeline
    X_train = pd.DataFrame({
        'city': ['A', 'B', 'A', 'C', 'B'] * 20, # Categorical
        'relevant_num': np.random.randint(1, 10, 100), # Numerical, relevant
        'irrelevant_num': np.random.rand(100) * 5, # Numerical, irrelevant
        'correlated_1': np.arange(100) * 2, # Highly correlated
        'correlated_2': np.arange(100) * 4 + np.random.normal(0, 5, 100),
        'useless_cat': ['X', 'Y'] * 50 # Categorical, irrelevant
    })
    # Add some missing values
    X_train.loc[5, 'relevant_num'] = np.nan
    X_train.loc[10, 'city'] = np.nan
    
    # Target is dependent on city and relevant_num, making them significant
    y_train = pd.Series( (X_train['city'].astype('category').cat.codes + X_train['relevant_num']) % 3 )

    # 1. Instantiate the transformation class
    transformation = DataTransformation()

    # 2. Get the pipeline object
    preprocessor_pipeline = transformation.get_data_preprocessor_object()

    # 3. Fit the pipeline on your training data
    print("\n--- Fitting and Transforming Data ---")
    X_train_transformed = preprocessor_pipeline.fit_transform(X_train, y_train)

    print("\n--- Results ---")
    print(f"Original data shape: {X_train.shape}")
    print(f"Transformed data shape: {X_train_transformed.shape}")
    print(f"\nFinal columns selected by the pipeline:\n{X_train_transformed.columns.tolist()}")