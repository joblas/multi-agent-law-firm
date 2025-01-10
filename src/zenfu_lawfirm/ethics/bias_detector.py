"""
Bias detection and mitigation module for the ZenFu Law Firm AI system.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class BiasDetector:
    def __init__(self):
        self.protected_attributes = ['gender', 'race', 'age', 'disability_status']
        self.metrics = {}

    def check_demographic_balance(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze demographic balance in the training data.
        
        Args:
            data: DataFrame containing the training data
            
        Returns:
            Dictionary containing balance metrics for each protected attribute
        """
        balance_metrics = {}
        for attribute in self.protected_attributes:
            if attribute in data.columns:
                distribution = data[attribute].value_counts(normalize=True)
                balance_metrics[attribute] = {
                    'distribution': distribution.to_dict(),
                    'entropy': -sum(p * np.log2(p) for p in distribution if p > 0)
                }
        return balance_metrics

    def analyze_representation(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze representation across different demographic groups.
        
        Args:
            data: DataFrame containing the dataset
            
        Returns:
            Dictionary containing representation metrics
        """
        representation_metrics = {}
        for attribute in self.protected_attributes:
            if attribute in data.columns:
                group_stats = data.groupby(attribute).agg({
                    'case_count': 'count',
                    'outcome': ['mean', 'std']
                }).to_dict()
                representation_metrics[attribute] = group_stats
        return representation_metrics

    def detect_historical_bias(self, 
                             historical_data: pd.DataFrame,
                             current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect historical bias by comparing past and current data distributions.
        
        Args:
            historical_data: DataFrame containing historical cases
            current_data: DataFrame containing current cases
            
        Returns:
            Dictionary containing bias metrics
        """
        bias_metrics = {}
        for attribute in self.protected_attributes:
            if attribute in historical_data.columns and attribute in current_data.columns:
                hist_dist = historical_data[attribute].value_counts(normalize=True)
                curr_dist = current_data[attribute].value_counts(normalize=True)
                
                # Calculate KL divergence
                kl_div = sum(
                    p * np.log2(p/q) 
                    for p, q in zip(curr_dist, hist_dist) 
                    if p > 0 and q > 0
                )
                bias_metrics[attribute] = {
                    'kl_divergence': kl_div,
                    'historical_distribution': hist_dist.to_dict(),
                    'current_distribution': curr_dist.to_dict()
                }
        return bias_metrics

    def check_fairness_metrics(self, 
                             predictions: np.ndarray,
                             actual: np.ndarray,
                             protected_attributes: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate fairness metrics for model predictions across protected groups.
        
        Args:
            predictions: Model predictions
            actual: Actual outcomes
            protected_attributes: DataFrame containing protected attribute information
            
        Returns:
            Dictionary containing fairness metrics for each protected attribute
        """
        fairness_metrics = {}
        
        for attribute in self.protected_attributes:
            if attribute in protected_attributes.columns:
                groups = protected_attributes[attribute].unique()
                group_metrics = {}
                
                for group in groups:
                    mask = protected_attributes[attribute] == group
                    group_preds = predictions[mask]
                    group_actual = actual[mask]
                    
                    tn, fp, fn, tp = confusion_matrix(
                        group_actual, group_preds, labels=[0, 1]
                    ).ravel()
                    
                    # Calculate metrics
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    group_metrics[group] = {
                        'false_positive_rate': fpr,
                        'false_negative_rate': fnr,
                        'true_positive_rate': tpr
                    }
                
                fairness_metrics[attribute] = group_metrics
        
        return fairness_metrics

    def analyze_disparate_impact(self, 
                               predictions: np.ndarray,
                               protected_attributes: pd.DataFrame,
                               threshold: float = 0.8) -> Dict[str, Dict]:
        """
        Analyze disparate impact of model predictions across protected groups.
        
        Args:
            predictions: Model predictions
            protected_attributes: DataFrame containing protected attribute information
            threshold: Threshold for determining disparate impact
            
        Returns:
            Dictionary containing disparate impact metrics
        """
        impact_metrics = {}
        
        for attribute in self.protected_attributes:
            if attribute in protected_attributes.columns:
                groups = protected_attributes[attribute].unique()
                group_rates = {}
                
                for group in groups:
                    mask = protected_attributes[attribute] == group
                    group_preds = predictions[mask]
                    positive_rate = np.mean(group_preds)
                    group_rates[group] = positive_rate
                
                # Calculate disparate impact ratios
                reference_rate = max(group_rates.values())
                impact_ratios = {
                    group: rate / reference_rate
                    for group, rate in group_rates.items()
                }
                
                impact_metrics[attribute] = {
                    'group_rates': group_rates,
                    'impact_ratios': impact_ratios,
                    'has_disparate_impact': any(ratio < threshold for ratio in impact_ratios.values())
                }
        
        return impact_metrics

    def verify_equal_opportunity(self, 
                               predictions: np.ndarray,
                               actual: np.ndarray,
                               protected_attributes: pd.DataFrame) -> Dict[str, Dict]:
        """
        Verify equal opportunity across protected groups.
        
        Args:
            predictions: Model predictions
            actual: Actual outcomes
            protected_attributes: DataFrame containing protected attribute information
            
        Returns:
            Dictionary containing equal opportunity metrics
        """
        opportunity_metrics = {}
        
        for attribute in self.protected_attributes:
            if attribute in protected_attributes.columns:
                groups = protected_attributes[attribute].unique()
                group_metrics = {}
                
                for group in groups:
                    mask = protected_attributes[attribute] == group
                    group_preds = predictions[mask]
                    group_actual = actual[mask]
                    
                    # Calculate true positive rate (equal opportunity metric)
                    positive_mask = group_actual == 1
                    if sum(positive_mask) > 0:
                        tpr = np.mean(group_preds[positive_mask])
                        group_metrics[group] = {'true_positive_rate': tpr}
                
                # Calculate differences in true positive rates
                tpr_values = [metrics['true_positive_rate'] 
                            for metrics in group_metrics.values()]
                max_difference = max(tpr_values) - min(tpr_values)
                
                opportunity_metrics[attribute] = {
                    'group_metrics': group_metrics,
                    'max_tpr_difference': max_difference,
                    'equal_opportunity_violation': max_difference > 0.1  # threshold of 0.1
                }
        
        return opportunity_metrics
