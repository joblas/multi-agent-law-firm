# src/zenfu_lawfirm/tools/policy_monitor.py
from typing import List, Dict
import requests
from datetime import datetime
import asyncio
import aiohttp
import logging
import re

class PolicyMonitor:
    """Monitor and analyze policy changes relevant to civil rights"""
    
    def __init__(self):
        self.tracked_sources = {
            'congress.gov': {
                'url': 'https://api.congress.gov/v3/bill',
                'type': 'legislation'
            },
            'supremecourt.gov': {
                'url': 'https://www.supremecourt.gov/opinions/opinions.aspx',
                'type': 'court_decisions'
            },
            'regulations.gov': {
                'url': 'https://api.regulations.gov',
                'type': 'regulations'
            }
        }
        self.categories = [
            'civil_rights',
            'discrimination',
            'privacy',
            'employment',
            'housing',
            'education'
        ]
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    async def monitor_policy_changes(self) -> List[Dict]:
        """Monitor for relevant policy and law changes"""
        updates = []
        tasks = []
        
        for source, info in self.tracked_sources.items():
            task = asyncio.create_task(self._fetch_updates(
                source, 
                info['url'],
                info['type']
            ))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching updates: {str(result)}")
            else:
                updates.extend(result)
        
        analyzed_updates = self._analyze_updates(updates)
        return self._prioritize_updates(analyzed_updates)
    
    async def _fetch_updates(self,
                           source: str,
                           url: str,
                           source_type: str) -> List[Dict]:
        """Fetch updates from a specific source"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status} from {source}")
                    data = await response.json()
                    return self._parse_updates(data, source, source_type)
        except Exception as e:
            self.logger.error(f"Error fetching updates from {source}: {str(e)}")
            return []
    
    def _parse_updates(self,
                      data: Dict,
                      source: str,
                      source_type: str) -> List[Dict]:
        """Parse updates based on source type"""
        if source_type == 'legislation':
            return self._parse_legislation(data)
        elif source_type == 'court_decisions':
            return self._parse_court_decisions(data)
        elif source_type == 'regulations':
            return self._parse_regulations(data)
        return []
    
    def _parse_legislation(self, data: Dict) -> List[Dict]:
        """Parse legislation updates"""
        updates = []
        bills = data.get('bills', [])
        
        for bill in bills:
            if self._is_relevant_to_civil_rights(bill.get('title', '')):
                updates.append({
                    'type': 'legislation',
                    'title': bill.get('title'),
                    'summary': bill.get('summary'),
                    'status': bill.get('status'),
                    'introduced_date': bill.get('introduced_date'),
                    'last_action_date': bill.get('last_action_date'),
                    'sponsors': bill.get('sponsors', []),
                    'categories': self._categorize_content(bill.get('title', '') + 
                                                        bill.get('summary', ''))
                })
        
        return updates
    
    def _parse_court_decisions(self, data: Dict) -> List[Dict]:
        """Parse court decision updates"""
        updates = []
        decisions = data.get('decisions', [])
        
        for decision in decisions:
            if self._is_relevant_to_civil_rights(decision.get('summary', '')):
                updates.append({
                    'type': 'court_decision',
                    'case_name': decision.get('case_name'),
                    'summary': decision.get('summary'),
                    'decision_date': decision.get('date'),
                    'majority_opinion': decision.get('majority_opinion'),
                    'dissenting_opinion': decision.get('dissenting_opinion'),
                    'categories': self._categorize_content(decision.get('summary', ''))
                })
        
        return updates
    
    def _parse_regulations(self, data: Dict) -> List[Dict]:
        """Parse regulation updates"""
        updates = []
        regulations = data.get('regulations', [])
        
        for regulation in regulations:
            if self._is_relevant_to_civil_rights(regulation.get('title', '')):
                updates.append({
                    'type': 'regulation',
                    'title': regulation.get('title'),
                    'agency': regulation.get('agency'),
                    'summary': regulation.get('summary'),
                    'status': regulation.get('status'),
                    'effective_date': regulation.get('effective_date'),
                    'categories': self._categorize_content(regulation.get('title', '') +
                                                        regulation.get('summary', ''))
                })
        
        return updates
    
    def _is_relevant_to_civil_rights(self, text: str) -> bool:
        """Check if the content is relevant to civil rights"""
        keywords = [
            'civil rights', 'discrimination', 'equal protection',
            'civil liberties', 'constitutional rights', 'human rights',
            'privacy rights', 'voting rights', 'workers rights',
            'fair housing', 'education equality', 'accessibility'
        ]
        
        text = text.lower()
        return any(keyword in text for keyword in keywords)
    
    def _categorize_content(self, text: str) -> List[str]:
        """Categorize content based on keywords"""
        text = text.lower()
        categories = []
        
        category_keywords = {
            'civil_rights': ['civil rights', 'constitutional', 'liberty'],
            'discrimination': ['discrimination', 'bias', 'prejudice'],
            'privacy': ['privacy', 'data protection', 'surveillance'],
            'employment': ['employment', 'workplace', 'labor'],
            'housing': ['housing', 'tenant', 'residential'],
            'education': ['education', 'school', 'academic']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _analyze_updates(self, updates: List[Dict]) -> List[Dict]:
        """Analyze the impact and relevance of updates"""
        analyzed_updates = []
        
        for update in updates:
            impact_analysis = self._assess_impact(update)
            relevance_score = self._calculate_relevance(update)
            
            if relevance_score > 0.5:  # Threshold for relevance
                analyzed_updates.append({
                    **update,
                    'impact_analysis': impact_analysis,
                    'relevance_score': relevance_score
                })
        
        return analyzed_updates
    
    def _assess_impact(self, update: Dict) -> Dict:
        """Assess the potential impact of an update"""
        # Calculate severity score based on various factors
        severity = 0
        
        # Check if it affects multiple categories
        if len(update.get('categories', [])) > 1:
            severity += 0.3
            
        # Check if it's from a high-priority source
        if update.get('type') == 'court_decision':
            severity += 0.4
        elif update.get('type') == 'legislation':
            severity += 0.3
            
        # Check for urgent keywords
        urgent_keywords = ['immediate', 'urgent', 'emergency', 'critical']
        if any(keyword in str(update).lower() for keyword in urgent_keywords):
            severity += 0.3
            
        return {
            'severity': min(severity, 1.0),
            'scope': len(update.get('categories', [])),
            'urgency': 'high' if severity > 0.7 else 'medium' if severity > 0.4 else 'low'
        }
    
    def _calculate_relevance(self, update: Dict) -> float:
        """Calculate relevance score for an update"""
        relevance = 0.0
        
        # Category relevance
        num_categories = len(update.get('categories', []))
        relevance += min(num_categories * 0.2, 0.6)
        
        # Source relevance
        if update.get('type') == 'court_decision':
            relevance += 0.3
        elif update.get('type') == 'legislation':
            relevance += 0.2
        elif update.get('type') == 'regulation':
            relevance += 0.1
            
        # Content relevance - check for strong civil rights keywords
        strong_keywords = ['fundamental rights', 'constitutional violation', 'civil liberties']
        if any(keyword in str(update).lower() for keyword in strong_keywords):
            relevance += 0.2
            
        return min(relevance, 1.0)
    
    def _prioritize_updates(self, updates: List[Dict]) -> List[Dict]:
        """Prioritize updates based on impact and relevance"""
        return sorted(
            updates,
            key=lambda x: (x['impact_analysis']['severity'], x['relevance_score']),
            reverse=True
        )