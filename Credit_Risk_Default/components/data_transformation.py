import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

from Credit_Risk_Default.constants.training_pipeline import TARGET_COLUMN

from Credit_Risk_Default.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from Credit_Risk_Default.entity.config_entity import DataTransformationConfig
from Credit_Risk_Default.exception.exception import CreditDefaultException 
from Credit_Risk_Default.logging.logger import logging
from Credit_Risk_Default.utils.main_utils.utils import save_numpy_array_data,save_object

class Preprocessor():
    def __init__(self, chi2_p_value_threshold=0.05, vif_threshold=6.0):
        self.chi2_p_value_threshold = chi2_p_value_threshold
        self.vif_threshold = vif_threshold

        self.categorical_cols_ = []
        self.numerical_cols_ = []
        self.final_features_ = []

    def fit(self, X, y):
        logging.info("Fitting the ComprehensivePreprocessor.")
        X_processed = X.copy()

        # Step 1: Identify column types
        self.categorical_cols_ = [col for col in X.select_dtypes(include=['object', 'category']).columns if col != TARGET_COLUMN]
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        # Step 2: Chi-Square Tests for Feature Selection
        categorical_selected_features = []
        for feature in self.categorical_cols_:
            contingency_table = pd.crosstab(X_processed[feature], y)
            _, p_val, _, _ = chi2_contingency(contingency_table)
            if p_val <= self.chi2_p_value_threshold:
                categorical_selected_features.append(feature)

        # Step 4: VIF for Multicollinearity 
        vif_data = X_processed[self.numerical_cols_]
        total_no_of_numfeat = vif_data.shape[1]
        columns_to_be_kept = []
        column_index = 0

        for i in range(total_no_of_numfeat):
            vif_value = variance_inflation_factor(vif_data, column_index)
            # print (column_index,'---',vif_value)

            if vif_value <= self.vif_threshold:
                columns_to_be_kept.append(self.numerical_cols_[i])
                column_index += 1
            
            else:
                vif_data = vif_data.drop(self.numerical_cols_[i], axis=1)
        
        # step 5: ANOVA
        columns_to_be_kept_numerical = []

        for i in columns_to_be_kept:
            a = list(X_processed[i])  
            b = list(y)  
            
            group_P1 = [value for value, group in zip(a, b) if group == 'P1']
            group_P2 = [value for value, group in zip(a, b) if group == 'P2']
            group_P3 = [value for value, group in zip(a, b) if group == 'P3']
            group_P4 = [value for value, group in zip(a, b) if group == 'P4']


            f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

            if p_value <= 0.05:
                columns_to_be_kept_numerical.append(i)

        self.final_features_ = categorical_selected_features + columns_to_be_kept_numerical
        logging.info(f"Preprocessor fitting complete. Final features selected: {self.final_features_}")

        return self
        
    def transform(self, X):
            logging.info("Transforming data using the fitted preprocessor.")
            X_processed = X.copy()
            X_processed = X_processed[self.final_features_]
            
            X_processed.loc[X_processed['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
            X_processed.loc[X_processed['EDUCATION'] == '12TH',['EDUCATION']]             = 2
            X_processed.loc[X_processed['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
            X_processed.loc[X_processed['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
            X_processed.loc[X_processed['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
            X_processed.loc[X_processed['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
            X_processed.loc[X_processed['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3

            X_processed['EDUCATION'] = X_processed['EDUCATION'].astype(int)

            X_encoded = pd.get_dummies(X_processed, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

            return X_encoded




class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CreditDefaultException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditDefaultException(e, sys)
        
    def get_data_transformer_object(cls)->Pipeline:

        logging.info("Entered get_data_trnasformer_object method of Trnasformation class")
        try:
            pipeline = Pipeline(
                steps=[
                    ('preprocessor', Preprocessor(
                        chi2_p_value_threshold=0.05, 
                        vif_threshold=6.0
                    ))
                ]
            )
            logging.info("Pipeline created successfully.")
            return pipeline
        except Exception as e:
            raise CreditDefaultException(e,sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessor_pipeline = self.get_data_transformer_object()
            preprocessor_object = preprocessor_pipeline.fit(input_feature_train_df, target_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise CreditDefaultException(e,sys)
