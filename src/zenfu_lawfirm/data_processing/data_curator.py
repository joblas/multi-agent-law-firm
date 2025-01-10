# src/zenfu_lawfirm/data_processing/data_curator.py
from typing import Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

class DataCurator:
    """Curate and preprocess legal data for training and analysis"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('legal-bert-base-uncased')
        self.model = AutoModel.from_pretrained('legal-bert-base-uncased')
        self.scaler = StandardScaler()
        
    def curate_dataset(self,
                      raw_data: List[Dict],
                      data_type: str) -> pd.DataFrame:
        """
        Curate and clean a dataset
        
        Args:
            raw_data: List of dictionaries containing raw data
            data_type: Type of data (e.g., 'case_law', 'policy', 'precedent')
            
        Returns:
            Curated pandas DataFrame
        """
        df = pd.DataFrame(raw_data)
        
        # Basic cleaning
        df = self._clean_text_data(df)
        df = self._handle_missing_values(df)
        df = self._standardize_formats(df)
        
        # Type-specific processing
        if data_type == 'case_law':
            df = self._process_case_law(df)
        elif data_type == 'policy':
            df = self._process_policy_data(df)
        elif data_type == 'precedent':
            df = self._process_precedent_data(df)
        
        # Final validation
        df = self._validate_dataset(df)
        
        return df
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text data"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            df[col] = df[col].apply(self._normalize_text)
        
        return df
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text data"""
        if pd.isna(text):
            return text
        
        text = text.lower()
        text = self._remove_special_characters(text)
        text = self._standardize_punctuation(text)
        
        return text
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        # Date formatting
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols)
        
        return df
