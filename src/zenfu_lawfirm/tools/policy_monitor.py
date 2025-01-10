# src/zenfu_lawfirm/tools/policy_monitor.py
from typing import List, Dict
import requests
from datetime import datetime
import asyncio
import logging

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
        
        results = await asyncio.gather(*tasks)
        for result in results:
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
                    data = await response.json()
                    return self._parse_updates(data, source, source_type)
        except Exception as e:
            logging.error(f"Error fetching updates from {source}: {str(e)}")
            return []
    
    def _parse_updates(self,
                      data: Dict,
                      source: str,
                      source_type: str) -> List[Dict]:
        """Parse updates based on source type"""
        updates = []
        
        if source_type == 'legislation':
            updates = self._parse_legislation(data)
        elif source_type == 'court_decisions':
            updates = self._parse_court_decisions(data)
        elif source_type == 'regulations':
            updates = self._parse_regulations(data)
        
        return updates
    
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
    
    def _prioritize_updates(self, updates: List[Dict]) -> List[Dict]:
        """Prioritize updates based on impact and relevance"""
        return sorted(
            updates,
            key=lambda x: (x['impact_analysis']['severity'], x['relevance_score']),
            reverse=True
        )
