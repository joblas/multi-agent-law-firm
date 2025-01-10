"""
Accountability and privacy protection module for the ZenFu Law Firm AI system.
"""

from typing import Dict, List, Any
import hashlib
import json
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

class PrivacyLevel(Enum):
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"

@dataclass
class AccessLog:
    timestamp: str
    user_id: str
    action: str
    resource_id: str
    privacy_level: PrivacyLevel
    success: bool
    reason: str = ""

class AccountabilityTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.access_logs = []
        self.privacy_policies = {}
        self.data_handling_protocols = {}
        self.sensitive_fields = set()

    def setup_privacy_policies(self, policies: Dict[str, Any]) -> None:
        """
        Set up privacy policies for different types of information.
        
        Args:
            policies: Dictionary containing privacy policies
        """
        self.privacy_policies = policies
        self.logger.info("Privacy policies updated")

    def register_sensitive_fields(self, fields: List[str]) -> None:
        """
        Register fields that contain sensitive information.
        
        Args:
            fields: List of field names containing sensitive information
        """
        self.sensitive_fields.update(fields)
        self.logger.info(f"Registered {len(fields)} sensitive fields")

    def anonymize_data(self, 
                      data: Dict[str, Any],
                      fields_to_anonymize: List[str] = None) -> Dict[str, Any]:
        """
        Anonymize sensitive information in the data.
        
        Args:
            data: Dictionary containing the data to anonymize
            fields_to_anonymize: Optional list of specific fields to anonymize
            
        Returns:
            Dictionary containing anonymized data
        """
        if fields_to_anonymize is None:
            fields_to_anonymize = self.sensitive_fields

        anonymized_data = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized_data:
                if isinstance(anonymized_data[field], str):
                    # Hash sensitive string values
                    anonymized_data[field] = hashlib.sha256(
                        anonymized_data[field].encode()
                    ).hexdigest()
                elif isinstance(anonymized_data[field], (int, float)):
                    # Mask numeric values
                    anonymized_data[field] = -999
                else:
                    # Remove other types of sensitive data
                    anonymized_data[field] = None

        return anonymized_data

    def encrypt_sensitive_data(self, 
                             data: Dict[str, Any],
                             encryption_key: bytes) -> Dict[str, Any]:
        """
        Encrypt sensitive information using provided encryption key.
        
        Args:
            data: Dictionary containing data to encrypt
            encryption_key: Encryption key
            
        Returns:
            Dictionary containing encrypted data
        """
        from cryptography.fernet import Fernet
        
        f = Fernet(encryption_key)
        encrypted_data = {}
        
        for field, value in data.items():
            if field in self.sensitive_fields:
                # Encrypt sensitive fields
                encrypted_data[field] = f.encrypt(
                    json.dumps(value).encode()
                ).decode()
            else:
                # Keep non-sensitive fields as is
                encrypted_data[field] = value
                
        return encrypted_data

    def log_access(self, 
                  user_id: str,
                  action: str,
                  resource_id: str,
                  privacy_level: PrivacyLevel,
                  success: bool,
                  reason: str = "") -> None:
        """
        Log access to sensitive information.
        
        Args:
            user_id: ID of the user accessing the information
            action: Type of action performed
            resource_id: ID of the accessed resource
            privacy_level: Privacy level of the accessed resource
            success: Whether the access was successful
            reason: Optional reason for access denial
        """
        log_entry = AccessLog(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            privacy_level=privacy_level,
            success=success,
            reason=reason
        )
        
        self.access_logs.append(log_entry)
        
        if not success:
            self.logger.warning(
                f"Access denied for user {user_id} to resource {resource_id}: {reason}"
            )

    def check_access_control(self, 
                           user_id: str,
                           resource_id: str,
                           action: str) -> bool:
        """
        Check if a user has permission to access a resource.
        
        Args:
            user_id: ID of the user requesting access
            resource_id: ID of the resource being accessed
            action: Type of action being requested
            
        Returns:
            Boolean indicating whether access is granted
        """
        # Implementation would depend on specific access control policies
        # This is a placeholder implementation
        if resource_id in self.privacy_policies:
            policy = self.privacy_policies[resource_id]
            
            if action not in policy['allowed_actions']:
                self.log_access(
                    user_id=user_id,
                    action=action,
                    resource_id=resource_id,
                    privacy_level=PrivacyLevel(policy['privacy_level']),
                    success=False,
                    reason=f"Action {action} not allowed"
                )
                return False
                
            if user_id not in policy['authorized_users']:
                self.log_access(
                    user_id=user_id,
                    action=action,
                    resource_id=resource_id,
                    privacy_level=PrivacyLevel(policy['privacy_level']),
                    success=False,
                    reason="User not authorized"
                )
                return False
                
            self.log_access(
                user_id=user_id,
                action=action,
                resource_id=resource_id,
                privacy_level=PrivacyLevel(policy['privacy_level']),
                success=True
            )
            return True
            
        return False

    def handle_personal_identifiers(self, 
                                  data: Dict[str, Any],
                                  identifier_fields: List[str]) -> Dict[str, Any]:
        """
        Handle personal identifiers according to privacy policies.
        
        Args:
            data: Dictionary containing data with personal identifiers
            identifier_fields: List of fields containing personal identifiers
            
        Returns:
            Dictionary with properly handled personal identifiers
        """
        handled_data = data.copy()
        
        for field in identifier_fields:
            if field in handled_data:
                if field in self.privacy_policies:
                    policy = self.privacy_policies[field]
                    
                    if policy['handling'] == 'anonymize':
                        handled_data = self.anonymize_data(
                            handled_data, [field]
                        )
                    elif policy['handling'] == 'encrypt':
                        # In practice, encryption key would be securely managed
                        handled_data = self.encrypt_sensitive_data(
                            handled_data,
                            encryption_key=b'dummy_key_for_example'
                        )
                    elif policy['handling'] == 'remove':
                        del handled_data[field]
                        
        return handled_data

    def protect_medical_information(self, 
                                  data: Dict[str, Any],
                                  medical_fields: List[str]) -> Dict[str, Any]:
        """
        Apply special protection to medical information.
        
        Args:
            data: Dictionary containing medical information
            medical_fields: List of fields containing medical information
            
        Returns:
            Dictionary with protected medical information
        """
        protected_data = data.copy()
        
        for field in medical_fields:
            if field in protected_data:
                # Always encrypt medical information
                protected_data = self.encrypt_sensitive_data(
                    protected_data,
                    encryption_key=b'dummy_key_for_example'
                )
                
                # Log access to medical information
                self.log_access(
                    user_id="system",
                    action="protect_medical_info",
                    resource_id=field,
                    privacy_level=PrivacyLevel.SENSITIVE,
                    success=True
                )
                
        return protected_data

    def generate_accountability_report(self, 
                                    start_date: str = None,
                                    end_date: str = None) -> Dict[str, Any]:
        """
        Generate a report of all access logs and privacy measures.
        
        Args:
            start_date: Optional start date for filtering (ISO format)
            end_date: Optional end date for filtering (ISO format)
            
        Returns:
            Dictionary containing the accountability report
        """
        filtered_logs = []
        
        for log in self.access_logs:
            log_date = datetime.fromisoformat(log.timestamp)
            
            if start_date:
                start = datetime.fromisoformat(start_date)
                if log_date < start:
                    continue
                    
            if end_date:
                end = datetime.fromisoformat(end_date)
                if log_date > end:
                    continue
                    
            filtered_logs.append({
                'timestamp': log.timestamp,
                'user_id': log.user_id,
                'action': log.action,
                'resource_id': log.resource_id,
                'privacy_level': log.privacy_level.value,
                'success': log.success,
                'reason': log.reason
            })
            
        return {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'accountability_report',
            'access_logs': filtered_logs,
            'privacy_policies': self.privacy_policies,
            'sensitive_fields': list(self.sensitive_fields),
            'total_access_attempts': len(filtered_logs),
            'denied_access_count': sum(
                1 for log in filtered_logs if not log['success']
            )
        }
