import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import re

class BiasMitigation:
    """Handle bias detection and mitigation in legal data processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.scaler = StandardScaler()
        
    def mitigate_dataset_bias(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """
        Mitigate bias in the dataset by applying various techniques
        
        Args:
            df: Input DataFrame
            sensitive_columns: List of columns containing sensitive attributes
            
        Returns:
            DataFrame with mitigated bias
        """
        try:
            # Create copy to avoid modifying original
            df_mitigated = df.copy()
            
            # Apply reweighting to balance sensitive attributes
            df_mitigated = self._apply_reweighting(df_mitigated, sensitive_columns)
            
            # Apply feature transformation to reduce bias
            df_mitigated = self._transform_features(df_mitigated, sensitive_columns)
            
            # Validate results
            self._validate_mitigation(df, df_mitigated, sensitive_columns)
            
            return df_mitigated
            
        except Exception as e:
            self.logger.error(f"Error in mitigate_dataset_bias: {str(e)}")
            return df
            
    def _apply_reweighting(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """Apply reweighting to balance sensitive attributes"""
        try:
            for col in sensitive_columns:
                if col in df.columns:
                    # Calculate value counts and weights
                    value_counts = df[col].value_counts()
                    max_count = value_counts.max()
                    weights = {val: max_count/count for val, count in value_counts.items()}
                    
                    # Apply weights
                    df['weight'] = df[col].map(weights)
                    
                    # Normalize weights
                    df['weight'] = df['weight'] / df['weight'].sum()
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _apply_reweighting: {str(e)}")
            return df
            
    def _transform_features(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """Transform features to reduce bias"""
        try:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            
            # Scale numeric features
            for col in numeric_columns:
                if col not in sensitive_columns and col != 'weight':
                    df[col] = self.scaler.fit_transform(df[[col]])
                    
            # One-hot encode categorical features
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in sensitive_columns:
                    df = pd.get_dummies(df, columns=[col], prefix=col)
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _transform_features: {str(e)}")
            return df
            
    def _validate_mitigation(self, df_original: pd.DataFrame, df_mitigated: pd.DataFrame, 
                           sensitive_columns: List[str]) -> None:
        """Validate bias mitigation results"""
        try:
            for col in sensitive_columns:
                if col in df_original.columns and col in df_mitigated.columns:
                    # Compare distributions
                    orig_dist = df_original[col].value_counts(normalize=True)
                    mit_dist = df_mitigated[col].value_counts(normalize=True)
                    
                    # Calculate distribution difference
                    dist_diff = abs(orig_dist - mit_dist).mean()
                    
                    if dist_diff > 0.1:  # Threshold for significant difference
                        self.logger.warning(
                            f"Large distribution difference detected in {col}: {dist_diff:.3f}"
                        )
                        
        except Exception as e:
            self.logger.error(f"Error in _validate_mitigation: {str(e)}")
            
    def measure_bias(self, df: pd.DataFrame, sensitive_columns: List[str], 
                    target_column: str) -> Dict[str, float]:
        """
        Measure bias metrics for sensitive attributes
        
        Args:
            df: Input DataFrame
            sensitive_columns: List of columns containing sensitive attributes
            target_column: Target variable column name
            
        Returns:
            Dictionary of bias metrics
        """
        try:
            metrics = {}
            
            for col in sensitive_columns:
                if col in df.columns and target_column in df.columns:
                    # Calculate disparate impact
                    metrics[f'disparate_impact_{col}'] = self._calculate_disparate_impact(
                        df, col, target_column
                    )
                    
                    # Calculate equal opportunity difference
                    metrics[f'equal_opportunity_diff_{col}'] = self._calculate_equal_opportunity_diff(
                        df, col, target_column
                    )
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in measure_bias: {str(e)}")
            return {}
            
    def _calculate_disparate_impact(self, df: pd.DataFrame, sensitive_col: str, 
                                  target_col: str) -> float:
        """Calculate disparate impact metric"""
        try:
            # Get unique values in sensitive column
            groups = df[sensitive_col].unique()
            
            if len(groups) < 2:
                return 1.0
                
            # Calculate positive outcome rates for each group
            rates = {}
            for group in groups:
                group_df = df[df[sensitive_col] == group]
                rates[group] = (group_df[target_col] == 1).mean()
                
            # Calculate disparate impact
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            
            return min_rate / max_rate if max_rate > 0 else 1.0
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_disparate_impact: {str(e)}")
            return 1.0
            
    def _calculate_equal_opportunity_diff(self, df: pd.DataFrame, sensitive_col: str, 
                                        target_col: str) -> float:
        """Calculate equal opportunity difference metric"""
        try:
            # Get unique values in sensitive column
            groups = df[sensitive_col].unique()
            
            if len(groups) < 2:
                return 0.0
                
            # Calculate true positive rates for each group
            tprs = {}
            for group in groups:
                group_df = df[df[sensitive_col] == group]
                true_positives = ((group_df[target_col] == 1) & 
                                (group_df[target_col] == 1)).sum()
                actual_positives = (group_df[target_col] == 1).sum()
                
                tprs[group] = (true_positives / actual_positives 
                              if actual_positives > 0 else 0.0)
                
            # Calculate difference between max and min TPR
            return max(tprs.values()) - min(tprs.values())
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_equal_opportunity_diff: {str(e)}")
            return 0.0
