from typing import List, Dict, Optional
import asyncio
import logging
from .document_tool import DocumentTool
from .policy_monitor import PolicyMonitor

class LegalToolManager:
    """Manages and coordinates legal analysis tools"""
    
    def __init__(self, docs_path: str):
        """
        Initialize the legal tool manager
        
        Args:
            docs_path: Path to the directory containing legal documents
        """
        self.doc_tool = DocumentTool(docs_path)
        self.policy_monitor = PolicyMonitor()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    async def analyze_case(self, 
                          case_text: str,
                          check_policies: bool = True) -> Dict:
        """
        Analyze a case using both document search and policy monitoring
        
        Args:
            case_text: The text of the case to analyze
            check_policies: Whether to check for relevant policy updates
            
        Returns:
            Dict containing analysis results and relevant updates
        """
        results = {
            'document_analysis': [],
            'policy_updates': [],
            'combined_insights': []
        }
        
        try:
            # Document analysis
            doc_results = self.doc_tool.search_documents(case_text)
            results['document_analysis'] = doc_results
            
            # Policy monitoring if requested
            if check_policies:
                policy_updates = await self.policy_monitor.monitor_policy_changes()
                relevant_updates = self._filter_relevant_updates(policy_updates, case_text)
                results['policy_updates'] = relevant_updates
            
            # Combine insights
            results['combined_insights'] = self._synthesize_analysis(
                doc_results,
                results['policy_updates']
            )
            
        except Exception as e:
            self.logger.error(f"Error in case analysis: {str(e)}")
        
        return results
    
    def add_document(self,
                    content: str,
                    metadata: Optional[Dict] = None) -> bool:
        """
        Add a new document to the search index
        
        Args:
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            bool: Success status
        """
        try:
            return self.doc_tool.add_document(content, metadata)
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return False
    
    def _filter_relevant_updates(self,
                               updates: List[Dict],
                               case_text: str) -> List[Dict]:
        """
        Filter policy updates relevant to the case
        
        Args:
            updates: List of policy updates
            case_text: Text of the current case
            
        Returns:
            List of relevant policy updates
        """
        relevant_updates = []
        case_categories = self.policy_monitor._categorize_content(case_text)
        
        for update in updates:
            update_categories = set(update.get('categories', []))
            if update_categories.intersection(case_categories):
                relevant_updates.append(update)
        
        return relevant_updates
    
    def _synthesize_analysis(self,
                           doc_results: List[Dict],
                           policy_updates: List[Dict]) -> List[Dict]:
        """
        Synthesize insights from both document and policy analysis
        
        Args:
            doc_results: Results from document analysis
            policy_updates: Relevant policy updates
            
        Returns:
            List of combined insights
        """
        insights = []
        
        # Document-based insights
        for result in doc_results:
            insights.append({
                'type': 'precedent',
                'source': result.get('metadata', {}).get('filename', 'Unknown'),
                'relevance': result.get('score', 0),
                'content': result.get('text', ''),
                'importance': 'high' if result.get('score', 0) > 0.8 else 'medium'
            })
        
        # Policy-based insights
        for update in policy_updates:
            insights.append({
                'type': 'policy_update',
                'source': update.get('type', 'Unknown'),
                'relevance': update.get('relevance_score', 0),
                'content': update.get('summary', ''),
                'importance': update.get('impact_analysis', {}).get('urgency', 'low')
            })
        
        # Sort by importance and relevance
        return sorted(
            insights,
            key=lambda x: (
                {'high': 2, 'medium': 1, 'low': 0}[x['importance']],
                x['relevance']
            ),
            reverse=True
        )
    
    async def get_policy_updates(self) -> List[Dict]:
        """
        Get latest policy updates
        
        Returns:
            List of policy updates
        """
        try:
            return await self.policy_monitor.monitor_policy_changes()
        except Exception as e:
            self.logger.error(f"Error getting policy updates: {str(e)}")
            return []
    
    def get_document_metadata(self) -> List[Dict]:
        """
        Get metadata for all indexed documents
        
        Returns:
            List of document metadata
        """
        try:
            return self.doc_tool.get_document_metadata()
        except Exception as e:
            self.logger.error(f"Error getting document metadata: {str(e)}")
            return []

def main():
    """Test the legal tool manager"""
    async def test():
        # Initialize manager
        manager = LegalToolManager(docs_path='legal_docs')
        
        # Add a test document
        manager.add_document(
            "This is a test document about civil rights violations.",
            metadata={
                'filename': 'test.txt',
                'categories': ['civil_rights']
            }
        )
        
        # Test case analysis
        results = await manager.analyze_case(
            "Looking for information about civil rights violations in employment."
        )
        
        print("Analysis Results:")
        print(results)
    
    # Run the test
    asyncio.run(test())

if __name__ == '__main__':
    main()
