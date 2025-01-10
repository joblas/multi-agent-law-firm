from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import re
from datetime import datetime

class DataCurator:
    """Curate and preprocess legal data for training and analysis"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('legal-bert-base-uncased')
        self.model = AutoModel.from_pretrained('legal-bert-base-uncased')
        self.scaler = StandardScaler()
        
    def curate_dataset(self, raw_data: List[Dict], data_type: str) -> pd.DataFrame:
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
        
        # Basic text normalization
        text = text.lower()
        text = self._remove_special_characters(text)
        text = self._standardize_punctuation(text)
        
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving legal citations"""
        if pd.isna(text):
            return text
        
        # Preserve legal citations (e.g., "123 F.2d 456")
        citation_pattern = r'\b\d+\s+F\.\d+d\s+\d+\b'
        citations = re.findall(citation_pattern, text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s\.\d]', ' ', text)
        
        # Restore citations
        for citation in citations:
            text = text.replace(citation.lower(), citation)
            
        return ' '.join(text.split())
    
    def _standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation in legal text"""
        if pd.isna(text):
            return text
        
        # Standardize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[\'']", "'", text)
        
        # Standardize ellipsis
        text = re.sub(r'\.{2,}', '...', text)
        
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
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols)
        
        return df
    
    def _process_case_law(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process case law specific data"""
        # Add case law specific features
        if 'decision_text' in df.columns:
            df['text_length'] = df['decision_text'].str.len()
            df['citation_count'] = df['decision_text'].apply(self._count_citations)
            
        # Extract and standardize case outcomes
        if 'outcome' in df.columns:
            df['outcome'] = df['outcome'].apply(self._standardize_outcome)
            
        return df
    
    def _process_policy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process policy specific data"""
        # Add policy specific features
        if 'policy_text' in df.columns:
            df['policy_length'] = df['policy_text'].str.len()
            df['effective_date'] = pd.to_datetime(df['effective_date'], errors='coerce')
            
        # Extract policy categories
        if 'policy_category' in df.columns:
            df = pd.get_dummies(df, columns=['policy_category'], prefix='category')
            
        return df
    
    def _process_precedent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process legal precedent specific data"""
        # Add precedent specific features
        if 'precedent_text' in df.columns:
            df['precedent_length'] = df['precedent_text'].str.len()
            df['citation_count'] = df['precedent_text'].apply(self._count_citations)
            
        # Extract jurisdictions
        if 'jurisdiction' in df.columns:
            df = pd.get_dummies(df, columns=['jurisdiction'], prefix='jurisdiction')
            
        return df
    
    def _count_citations(self, text: str) -> int:
        """Count legal citations in text"""
        if pd.isna(text):
            return 0
            
        citation_pattern = r'\b\d+\s+F\.\d+d\s+\d+\b'
        citations = re.findall(citation_pattern, text)
        return len(citations)
    
    def _standardize_outcome(self, outcome: str) -> str:
        """Standardize case outcomes"""
        if pd.isna(outcome):
            return 'unknown'
            
        outcome = outcome.lower().strip()
        
        # Map common outcome variations
        outcome_map = {
            'affirmed': 'affirmed',
            'reversed': 'reversed',
            'remanded': 'remanded',
            'dismissed': 'dismissed',
            'settled': 'settled'
        }
        
        return outcome_map.get(outcome, 'other')
    
    def _validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the final dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Check for remaining nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("Warning: Dataset contains null values:", null_counts[null_counts > 0])
        
        # Verify data types
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"Warning: Column {col} remains as object type")
                
        return df
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using the legal BERT model"""
        if pd.isna(text):
            return np.zeros(self.model.config.hidden_size)
            
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()
    
    def batch_encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        encodings = [self.encode_text(text) for text in texts]
        return np.vstack(encodings)