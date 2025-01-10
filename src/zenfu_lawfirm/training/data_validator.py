import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

class DataValidator:
    """Validates and prepares data for model training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Define validation thresholds
        self.min_samples = 100
        self.max_missing_ratio = 0.2
        self.min_feature_variance = 0.01
        
    def validate_training_data(self, df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Validate training data for quality and completeness
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Tuple containing:
            - bool: Whether validation passed
            - str: Validation message
            - Optional[pd.DataFrame]: Processed DataFrame if validation passed, None otherwise
        """
        try:
            validation_steps = [
                self._validate_sample_size,
                self._validate_feature_completeness,
                self._validate_feature_variance,
                self._validate_data_balance,
                self._validate_data_leakage
            ]
            
            for step in validation_steps:
                is_valid, message, df = step(df)
                if not is_valid:
                    return False, message, None
                    
            return True, "Data validation passed successfully", df
            
        except Exception as e:
            self.logger.error(f"Error in validate_training_data: {str(e)}")
            return False, f"Validation failed with error: {str(e)}", None
    
    def _validate_sample_size(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Validate if dataset has sufficient samples"""
        if len(df) < self.min_samples:
            return False, f"Insufficient samples: {len(df)} < {self.min_samples}", df
        return True, "Sample size validation passed", df
    
    def _validate_feature_completeness(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Validate feature completeness and handle missing values"""
        missing_ratio = df.isnull().sum() / len(df)
        
        # Check for columns with too many missing values
        problematic_cols = missing_ratio[missing_ratio > self.max_missing_ratio].index.tolist()
        if problematic_cols:
            return False, f"Excessive missing values in columns: {problematic_cols}", df
            
        return True, "Feature completeness validation passed", df
    
    def _validate_feature_variance(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Validate feature variance to detect constant or near-constant features"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        low_variance_cols = []
        for col in numeric_cols:
            if df[col].var() < self.min_feature_variance:
                low_variance_cols.append(col)
                
        if low_variance_cols:
            # Remove low variance columns
            df = df.drop(columns=low_variance_cols)
            self.logger.warning(f"Removed low variance columns: {low_variance_cols}")
            
        return True, "Feature variance validation passed", df
    
    def _validate_data_balance(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Validate class balance for classification tasks"""
        if 'label' in df.columns:
            class_counts = df['label'].value_counts()
            class_ratios = class_counts / len(df)
            
            # Check for severe class imbalance
            if any(ratio < 0.1 for ratio in class_ratios):
                self.logger.warning("Severe class imbalance detected")
                # Apply class weights or sampling techniques if needed
                
        return True, "Data balance validation passed", df
    
    def _validate_data_leakage(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Check for potential data leakage"""
        # Remove direct identifiers
        identifier_cols = [col for col in df.columns if 'id' in col.lower()]
        if identifier_cols:
            df = df.drop(columns=identifier_cols)
            self.logger.info(f"Removed identifier columns: {identifier_cols}")
            
        # Check for timestamp-related features
        time_cols = [col for col in df.columns if any(t in col.lower() for t in ['time', 'date'])]
        for col in time_cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Convert to relative time features
                reference_date = df[col].min()
                df[f'{col}_days'] = (df[col] - reference_date).dt.total_seconds() / (24 * 3600)
                df = df.drop(columns=[col])
                
        return True, "Data leakage validation passed", df
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2, 
                               validation_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Prepare train/validation/test splits
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        try:
            # First split: separate test set
            train_val, test = train_test_split(df, test_size=test_size, random_state=42)
            
            # Second split: separate validation set from training set
            val_size_adjusted = validation_size / (1 - test_size)
            train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42)
            
            splits = {
                'train': train,
                'validation': val,
                'test': test
            }
            
            # Log split sizes
            for split_name, split_df in splits.items():
                self.logger.info(f"{split_name} split size: {len(split_df)}")
                
            return splits
            
        except Exception as e:
            self.logger.error(f"Error in prepare_train_test_split: {str(e)}")
            raise
    
    def validate_feature_engineering(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """
        Validate feature engineering results
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple containing validation status, message, and processed DataFrame
        """
        try:
            # Check for infinite values
            inf_cols = df.columns[np.isinf(df).any()].tolist()
            if inf_cols:
                df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
                self.logger.warning(f"Replaced infinite values in columns: {inf_cols}")
            
            # Check feature correlations
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            high_corr_pairs.append((numeric_cols[i], numeric_cols[j]))
                
                if high_corr_pairs:
                    self.logger.warning(f"High correlation detected between features: {high_corr_pairs}")
            
            return True, "Feature engineering validation passed", df
            
        except Exception as e:
            self.logger.error(f"Error in validate_feature_engineering: {str(e)}")
            return False, f"Feature engineering validation failed: {str(e)}", df
