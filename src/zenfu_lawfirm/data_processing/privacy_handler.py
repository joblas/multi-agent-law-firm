import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import logging
import hashlib
import re
from datetime import datetime
import json

class PrivacyHandler:
    """Handle privacy concerns and data protection in legal data processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Common patterns for sensitive data
        self._pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b\d{4}[-]?\d{4}[-]?\d{4}[-]?\d{4}\b',
            'date_of_birth': r'\b\d{2}[-/]\d{2}[-/]\d{4}\b'
        }
        
    def anonymize_data(self, df: pd.DataFrame, sensitive_columns: List[str], 
                      method: str = 'hash') -> pd.DataFrame:
        """
        Anonymize sensitive data in the DataFrame
        
        Args:
            df: Input DataFrame
            sensitive_columns: List of columns containing sensitive data
            method: Anonymization method ('hash', 'mask', or 'categorize')
            
        Returns:
            DataFrame with anonymized data
        """
        try:
            df_anon = df.copy()
            
            for col in sensitive_columns:
                if col in df_anon.columns:
                    if method == 'hash':
                        df_anon[col] = df_anon[col].apply(self._hash_value)
                    elif method == 'mask':
                        df_anon[col] = df_anon[col].apply(self._mask_value)
                    elif method == 'categorize':
                        df_anon[col] = self._categorize_values(df_anon[col])
                    else:
                        self.logger.warning(f"Unknown anonymization method: {method}")
                        
            return df_anon
            
        except Exception as e:
            self.logger.error(f"Error in anonymize_data: {str(e)}")
            return df
            
    def detect_pii(self, df: pd.DataFrame) -> Set[str]:
        """
        Detect columns that might contain PII
        
        Args:
            df: Input DataFrame
            
        Returns:
            Set of column names that might contain PII
        """
        try:
            pii_columns = set()
            
            for col in df.columns:
                # Check column name for PII indicators
                if any(indicator in col.lower() for indicator in 
                      ['name', 'email', 'phone', 'address', 'ssn', 'birth']):
                    pii_columns.add(col)
                    continue
                
                # Sample values for pattern matching
                sample = df[col].dropna().astype(str).head(100)
                
                # Check for PII patterns
                for pattern_name, pattern in self._pii_patterns.items():
                    if any(bool(re.search(pattern, str(val))) for val in sample):
                        pii_columns.add(col)
                        self.logger.info(f"Detected potential {pattern_name} in column: {col}")
                        break
                        
            return pii_columns
            
        except Exception as e:
            self.logger.error(f"Error in detect_pii: {str(e)}")
            return set()
            
    def _hash_value(self, value: any) -> str:
        """Hash a value using SHA-256"""
        try:
            if pd.isna(value):
                return value
            return hashlib.sha256(str(value).encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error in _hash_value: {str(e)}")
            return value
            
    def _mask_value(self, value: any) -> str:
        """Mask a value, preserving some structure"""
        try:
            if pd.isna(value):
                return value
                
            value_str = str(value)
            
            # Handle different types of data
            if re.match(self._pii_patterns['email'], value_str):
                # Mask email while preserving domain
                username, domain = value_str.split('@')
                return f"{'*' * len(username)}@{domain}"
                
            elif re.match(self._pii_patterns['phone'], value_str):
                # Mask phone number while preserving last 4 digits
                clean_num = re.sub(r'[-.]', '', value_str)
                return f"{'*' * 6}{clean_num[-4:]}"
                
            elif re.match(self._pii_patterns['ssn'], value_str):
                # Mask SSN while preserving last 4 digits
                clean_ssn = re.sub(r'[-]', '', value_str)
                return f"{'*' * 5}{clean_ssn[-4:]}"
                
            else:
                # Default masking for other types
                return '*' * len(value_str)
                
        except Exception as e:
            self.logger.error(f"Error in _mask_value: {str(e)}")
            return value
            
    def _categorize_values(self, series: pd.Series) -> pd.Series:
        """Categorize values into groups"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                # Create categories based on percentiles
                return pd.qcut(series, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            else:
                # Create categories based on frequency
                value_counts = series.value_counts()
                categories = pd.qcut(value_counts, q=5, labels=['VeryRare', 'Rare', 'Medium', 'Common', 'VeryCommon'])
                return series.map(value_counts).map(lambda x: categories[x])
                
        except Exception as e:
            self.logger.error(f"Error in _categorize_values: {str(e)}")
            return series
            
    def encrypt_sensitive_fields(self, data: Dict, sensitive_fields: List[str]) -> Dict:
        """
        Encrypt sensitive fields in a dictionary
        
        Args:
            data: Input dictionary
            sensitive_fields: List of sensitive field names
            
        Returns:
            Dictionary with encrypted sensitive fields
        """
        try:
            encrypted_data = data.copy()
            
            for field in sensitive_fields:
                if field in encrypted_data:
                    encrypted_data[field] = self._hash_value(encrypted_data[field])
                    
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Error in encrypt_sensitive_fields: {str(e)}")
            return data
            
    def redact_sensitive_text(self, text: str) -> str:
        """
        Redact sensitive information from text
        
        Args:
            text: Input text
            
        Returns:
            Text with sensitive information redacted
        """
        try:
            redacted_text = text
            
            # Redact each type of PII
            for pattern_name, pattern in self._pii_patterns.items():
                redacted_text = re.sub(
                    pattern,
                    f"[REDACTED_{pattern_name.upper()}]",
                    redacted_text
                )
                
            return redacted_text
            
        except Exception as e:
            self.logger.error(f"Error in redact_sensitive_text: {str(e)}")
            return text
            
    def generate_privacy_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a privacy analysis report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing privacy analysis results
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'potential_pii_columns': list(self.detect_pii(df)),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_value_counts': {
                    col: df[col].nunique()
                    for col in df.columns
                },
                'recommendations': []
            }
            
            # Add recommendations based on analysis
            for col in report['potential_pii_columns']:
                report['recommendations'].append({
                    'column': col,
                    'recommendation': 'Consider anonymizing this column as it may contain PII'
                })
                
            # Check for columns with low uniqueness
            for col, unique_count in report['unique_value_counts'].items():
                if unique_count == len(df):
                    report['recommendations'].append({
                        'column': col,
                        'recommendation': 'This column has unique values for each row and might be an identifier'
                    })
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Error in generate_privacy_report: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
