"""
Transparency management module for the ZenFu Law Firm AI system.
"""

from typing import Dict, List, Any
import json
from datetime import datetime
import logging

class TransparencyManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_history = []
        self.documentation_records = {}

    def document_decision_rationale(self, 
                                  decision_id: str,
                                  rationale: str,
                                  factors: List[str],
                                  confidence_score: float) -> Dict[str, Any]:
        """
        Document the rationale behind a legal decision.
        
        Args:
            decision_id: Unique identifier for the decision
            rationale: Detailed explanation of the decision
            factors: List of factors considered in the decision
            confidence_score: Model's confidence score for the decision
            
        Returns:
            Dictionary containing the documented decision information
        """
        decision_doc = {
            'decision_id': decision_id,
            'timestamp': datetime.now().isoformat(),
            'rationale': rationale,
            'factors_considered': factors,
            'confidence_score': confidence_score,
        }
        
        self.decision_history.append(decision_doc)
        return decision_doc

    def add_precedent_citations(self, 
                              decision_id: str,
                              citations: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Add relevant legal precedent citations to a decision.
        
        Args:
            decision_id: Unique identifier for the decision
            citations: List of citation dictionaries containing case references
            
        Returns:
            Updated decision document
        """
        for decision in self.decision_history:
            if decision['decision_id'] == decision_id:
                decision['precedent_citations'] = [
                    {
                        'case_name': citation['case_name'],
                        'citation': citation['citation'],
                        'relevance': citation['relevance']
                    }
                    for citation in citations
                ]
                return decision
        
        raise ValueError(f"Decision with ID {decision_id} not found")

    def record_confidence_scores(self, 
                               decision_id: str,
                               scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Record confidence scores for different aspects of a decision.
        
        Args:
            decision_id: Unique identifier for the decision
            scores: Dictionary of confidence scores for different aspects
            
        Returns:
            Updated decision document
        """
        for decision in self.decision_history:
            if decision['decision_id'] == decision_id:
                decision['detailed_confidence_scores'] = scores
                decision['overall_confidence'] = sum(scores.values()) / len(scores)
                return decision
        
        raise ValueError(f"Decision with ID {decision_id} not found")

    def document_data_sources(self, 
                            sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Document the data sources used in the system.
        
        Args:
            sources: List of data source information dictionaries
            
        Returns:
            Updated documentation records
        """
        self.documentation_records['data_sources'] = [
            {
                'source_id': source['source_id'],
                'name': source['name'],
                'type': source['type'],
                'description': source['description'],
                'last_updated': source.get('last_updated', datetime.now().isoformat()),
                'validation_status': source.get('validation_status', 'pending')
            }
            for source in sources
        ]
        return self.documentation_records

    def document_model_limitations(self, 
                                 limitations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Document known limitations of the AI model.
        
        Args:
            limitations: List of limitation information dictionaries
            
        Returns:
            Updated documentation records
        """
        self.documentation_records['model_limitations'] = [
            {
                'limitation_id': limitation['limitation_id'],
                'category': limitation['category'],
                'description': limitation['description'],
                'potential_impact': limitation['potential_impact'],
                'mitigation_strategy': limitation.get('mitigation_strategy', 'Not provided')
            }
            for limitation in limitations
        ]
        return self.documentation_records

    def document_uncertainty_factors(self, 
                                  factors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Document factors contributing to uncertainty in decisions.
        
        Args:
            factors: List of uncertainty factor information dictionaries
            
        Returns:
            Updated documentation records
        """
        self.documentation_records['uncertainty_factors'] = [
            {
                'factor_id': factor['factor_id'],
                'name': factor['name'],
                'description': factor['description'],
                'impact_level': factor['impact_level'],
                'detection_method': factor.get('detection_method', 'Manual review'),
                'handling_strategy': factor.get('handling_strategy', 'Case-by-case evaluation')
            }
            for factor in factors
        ]
        return self.documentation_records

    def export_transparency_report(self, 
                                 decision_id: str = None,
                                 include_documentation: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive transparency report.
        
        Args:
            decision_id: Optional specific decision ID to report on
            include_documentation: Whether to include general documentation
            
        Returns:
            Dictionary containing the transparency report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'transparency_report'
        }

        if decision_id:
            decision = next(
                (d for d in self.decision_history if d['decision_id'] == decision_id),
                None
            )
            if decision:
                report['decision_details'] = decision
            else:
                raise ValueError(f"Decision with ID {decision_id} not found")
        else:
            report['decisions'] = self.decision_history

        if include_documentation:
            report['documentation'] = self.documentation_records

        return report

    def get_decision_history(self, 
                           start_date: str = None,
                           end_date: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve decision history within a specified date range.
        
        Args:
            start_date: Optional start date for filtering (ISO format)
            end_date: Optional end date for filtering (ISO format)
            
        Returns:
            List of decision documents within the specified range
        """
        if not (start_date or end_date):
            return self.decision_history

        filtered_history = []
        for decision in self.decision_history:
            decision_date = datetime.fromisoformat(decision['timestamp'])
            
            if start_date:
                start = datetime.fromisoformat(start_date)
                if decision_date < start:
                    continue
                    
            if end_date:
                end = datetime.fromisoformat(end_date)
                if decision_date > end:
                    continue
                    
            filtered_history.append(decision)
            
        return filtered_history
