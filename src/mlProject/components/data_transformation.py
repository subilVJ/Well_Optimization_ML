import os
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd
import numpy as np
from mlProject.utils.common import save_object
from mlProject.entity.config_entity import DataTransformationConfig




class DataTransfromation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config

    def train_test_spliting(self):
        data=pd.read_csv(self.config.data_path)

        ## Split data into train and split set (0.75,0.25)
        train,test=train_test_split(data)
        train.to_csv(self.config.train_data_path,index=False)
        test.to_csv(self.config.test_data_path,index=False)

        logger.info("Splitted data into train and test data")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
    
    def get_data_transformation(self):
        try:
            numerical_fetures=['permeability_md', 'porosity_fraction', 'net_to_gross', 'thickness_ft',
       'well_depth_ft', 'tubing_diameter_in', 'choke_size_64th',
       'reservoir_pressure_psi', 'reservoir_temp_f', 'bottomhole_pressure_psi',
       'wellhead_pressure_psi', 'oil_gravity_api', 'gas_oil_ratio_scf_bbl',
       'water_cut_fraction', 'fvf_oil', 'oil_viscosity_cp', 'oil_rate_bbl_day',
       'gas_rate_scf_day', 'water_rate_bbl_day', 'productivity_index',
       'oil_price_usd_bbl', 'gas_price_usd_mcf', 'daily_opex_usd',
       'drilling_cost_usd', 'completion_cost_usd', 'total_capex_usd',
       'daily_revenue_usd', 'oil_cut', 'profit_per_barrel',
       'production_efficiency', 'economic_efficiency', 'ranking_score',
       'well_age_days', 'production_months', 'days_since_workover',
       'pressure_drawdown', 'total_liquid_rate', 'productivity_factor']
            
            cat_features=['well_type', 'completion_type', 'artificial_lift',
       'depth_category']
            logger.info(f"Numerical Features {numerical_fetures}")
            logger.info(f"Categorical Features { cat_features}")


            num_transformer=StandardScaler()
            oh_transformer=OneHotEncoder()

            preprocessor=ColumnTransformer([
                ("OneHotEncoder",oh_transformer,cat_features),
                ("StandardScaler",num_transformer,numerical_fetures)
            ])

            return preprocessor
        except Exception as e:
            raise e

    def get_data_transformation(self):
        try:
            numerical_fetures=['permeability_md', 'porosity_fraction', 'net_to_gross', 'thickness_ft',
       'well_depth_ft', 'tubing_diameter_in', 'choke_size_64th',
       'reservoir_pressure_psi', 'reservoir_temp_f', 'bottomhole_pressure_psi',
       'wellhead_pressure_psi', 'oil_gravity_api', 'gas_oil_ratio_scf_bbl',
       'water_cut_fraction', 'fvf_oil', 'oil_viscosity_cp', 'oil_rate_bbl_day',
       'gas_rate_scf_day', 'water_rate_bbl_day', 'productivity_index',
       'oil_price_usd_bbl', 'gas_price_usd_mcf', 'daily_opex_usd',
       'drilling_cost_usd', 'completion_cost_usd', 'total_capex_usd',
       'daily_revenue_usd', 'oil_cut', 'profit_per_barrel',
       'production_efficiency', 'economic_efficiency', 'ranking_score',
       'well_age_days', 'production_months', 'days_since_workover',
       'pressure_drawdown', 'total_liquid_rate', 'productivity_factor']
            
            cat_features=['well_type', 'completion_type', 'artificial_lift',
       'depth_category']
            logger.info(f"Numerical Features {numerical_fetures}")
            logger.info(f"Categorical Features { cat_features}")


            num_transformer=StandardScaler()
            oh_transformer=OneHotEncoder()

            preprocessor=ColumnTransformer([
                ("OneHotEncoder",oh_transformer,cat_features),
                ("StandardScaler",num_transformer,numerical_fetures)
            ])

            return preprocessor
        except Exception as e:
            raise e

    def intiate_data_transfromation(self):
        try:
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logger.info("Reading train and test data completed")

            logger.info("Getting processing object")
            preprocessing_obj=self.get_data_transformation()

            target_column_name='performance_index'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info(f"Applying processsing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            train_data=pd.DataFrame(train_arr)
            test_data=pd.DataFrame(test_arr)

            train_data.to_csv(self.config.train_model_data,index=False)
            test_data.to_csv(self.config.test_model_data,index=False)
            

            save_object(
                file_path=self.config.preprocessor_obj_file_path,obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            raise e

