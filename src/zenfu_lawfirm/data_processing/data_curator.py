from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import re
from datetime import datetime
import logging

class DataCurator:
    """Curate and preprocess legal data for training and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        try:
            # Use standard BERT model instead of legal-specific
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.scaler = StandardScaler()
            self.logger.info("Successfully initialized DataCurator with BERT model")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            # Initialize without models if they fail to load
            self.tokenizer = None
            self.model = None
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
        try:
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
            else:
                self.logger.warning(f"Unknown data type: {data_type}")
            
            # Final validation
            df = self._validate_dataset(df)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in curate_dataset: {str(e)}")
            raise
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text data"""
        try:
            text_columns = df.select_dtypes(include=['object']).columns
            
            for col in text_columns:
                df[col] = df[col].apply(self._normalize_text)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in _clean_text_data: {str(e)}")
            return df
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text data"""
        if pd.isna(text):
            return text
        
        try:
            # Basic text normalization
            text = str(text).lower()
            text = self._remove_special_characters(text)
            text = self._standardize_punctuation(text)
            
            return text
        except Exception as e:
            self.logger.error(f"Error in _normalize_text: {str(e)}")
            return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving legal citations"""
        if pd.isna(text):
            return text
        
        try:
            # Preserve legal citations (e.g., "123 F.2d 456")
            citation_pattern = r'\b\d+\s+F\.\d+d\s+\d+\b'
            citations = re.findall(citation_pattern, text)
            
            # Remove special characters
            text = re.sub(r'[^\w\s\.\d]', ' ', text)
            
            # Restore citations
            for citation in citations:
                text = text.replace(citation.lower(), citation)
                
            return ' '.join(text.split())
        except Exception as e:
            self.logger.error(f"Error in _remove_special_characters: {str(e)}")
            return text
    
    def _standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation in legal text"""
        if pd.isna(text):
            return text
        
        try:
            # Standardize quotes
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r"[\'']", "'", text)
            
            # Standardize ellipsis
            text = re.sub(r'\.{2,}', '...', text)
            
            return text
        except Exception as e:
            self.logger.error(f"Error in _standardize_punctuation: {str(e)}")
            return text
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('unknown')
            
            return df
        except Exception as e:
            self.logger.error(f"Error in _handle_missing_values: {str(e)}")
            return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        try:
            # Date formatting
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Categorical encoding
            categorical_cols = df.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                df = pd.get_dummies(df, columns=categorical_cols)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in _standardize_formats: {str(e)}")
            return df
    
    def _process_case_law(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process case law specific data"""
        try:
            # Add case law specific features
            if 'decision_text' in df.columns:
                df['text_length'] = df['decision_text'].str.len()
                df['citation_count'] = df['decision_text'].apply(self._count_citations)
                
            # Extract and standardize case outcomes
            if 'outcome' in df.columns:
                df['outcome'] = df['outcome'].apply(self._standardize_outcome)
                
            return df
        except Exception as e:
            self.logger.error(f"Error in _process_case_law: {str(e)}")
            return df
    
    def _process_policy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process policy specific data"""
        try:
            # Add policy specific features
            if 'policy_text' in df.columns:
                df['policy_length'] = df['policy_text'].str.len()
                df['effective_date'] = pd.to_datetime(df['effective_date'], errors='coerce')
                
            # Extract policy categories
            if 'policy_category' in df.columns:
                df = pd.get_dummies(df, columns=['policy_category'], prefix='category')
                
            return df
        except Exception as e:
            self.logger.error(f"Error in _process_policy_data: {str(e)}")
            return df
    
    def _process_precedent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process legal precedent specific data"""
        try:
            # Add precedent specific features
            if 'precedent_text' in df.columns:
                df['precedent_length'] = df['precedent_text'].str.len()
                df['citation_count'] = df['precedent_text'].apply(self._count_citations)
                
            # Extract jurisdictions
            if 'jurisdiction' in df.columns:
                df = pd.get_dummies(df, columns=['jurisdiction'], prefix='jurisdiction')
                
            return df
        except Exception as e:
            self.logger.error(f"Error in _process_precedent_data: {str(e)}")
            return df
    
    def _count_citations(self, text: str) -> int:
        """Count legal citations in text"""
        if pd.isna(text):
            return 0
            
        try:
            citation_pattern = r'\b\d+\s+F\.\d+d\s+\d+\b'
            citations = re.findall(citation_pattern, str(text))
            return len(citations)
        except Exception as e:
            self.logger.error(f"Error in _count_citations: {str(e)}")
            return 0
    
    def _standardize_outcome(self, outcome: str) -> str:
        """Standardize case outcomes"""
        if pd.isna(outcome):
            return 'unknown'
            
        try:
            outcome = str(outcome).lower().strip()
            
            # Map common outcome variations
            outcome_map = {
                'affirmed': 'affirmed',
                'reversed': 'reversed',
                'remanded': 'remanded',
                'dismissed': 'dismissed',
                'settled': 'settled'
            }
            
            return outcome_map.get(outcome, 'other')
        except Exception as e:
            self.logger.error(f"Error in _standardize_outcome: {str(e)}")
            return 'unknown'
    
    def _validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the final dataset"""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Check for remaining nulls
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning("Dataset contains null values:", null_counts[null_counts > 0])
            
            # Verify data types
            for col in df.columns:
                if df[col].dtype == 'object':
                    self.logger.warning(f"Column {col} remains as object type")
                    
            return df
        except Exception as e:
            self.logger.error(f"Error in _validate_dataset: {str(e)}")
            return df
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using the BERT model"""
        if pd.isna(text) or self.model is None or self.tokenizer is None:
            return np.zeros(768)  # Standard BERT hidden size
            
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            outputs = self.model(**inputs)
            
            # Use CLS token embedding
            return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()
        except Exception as e:
            self.logger.error(f"Error in encode_text: {str(e)}")
            return np.zeros(768)  # Standard BERT hidden size
    
    def batch_encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        try:
            encodings = [self.encode_text(text) for text in texts]
            return np.vstack(encodings)
        except Exception as e:
            self.logger.error(f"Error in batch_encode_texts: {str(e)}")
            return np.zeros((len(texts), 768))  # Standard BERT hidden size
