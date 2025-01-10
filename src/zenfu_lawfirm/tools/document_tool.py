# src/zenfu_lawfirm/tools/document_tool.py
from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from llama_index.core.schema import TextNode
from llama_index.core.readers import SimpleDirectoryReader
from typing import List, Dict, Optional, Union
import os
import logging
from datetime import datetime

# Set up logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class Document:
    """Represents a document with its text and metadata"""
    
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata or {}
        # Add timestamp if not provided
        if 'date_added' not in self.metadata:
            self.metadata['date_added'] = datetime.now().isoformat()

    def to_llama_document(self) -> LlamaDocument:
        """Convert to LlamaIndex Document format"""
        return LlamaDocument(
            text=self.text,
            metadata=self.metadata
        )

class DocumentTool:
    """Tool for searching and analyzing legal documents"""
    
    def __init__(self, docs_path: str):
        """
        Initialize the document search tool
        
        Args:
            docs_path: Path to the directory containing legal documents
        """
        self.docs_path = docs_path
        self.logger = logging.getLogger(__name__)
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the document index"""
        try:
            if os.path.exists(self.docs_path):
                reader = SimpleDirectoryReader(self.docs_path)
                documents = reader.load_data()
                self.index = VectorStoreIndex.from_documents(documents)
                self.logger.info(f"Initialized document index with {len(documents)} documents")
            else:
                self.index = None
                self.logger.warning(f"Document path {self.docs_path} does not exist")
        except Exception as e:
            self.logger.error(f"Error initializing document index: {str(e)}")
            self.index = None

    def search_documents(self, 
                        query: str, 
                        top_k: int = 5,
                        min_score: float = 0.3) -> List[Dict]:
        """
        Search documents for relevant information
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant document snippets with metadata
        """
        try:
            if not self.index:
                return []
                
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)
            
            filtered_results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if node.score >= min_score:
                        filtered_results.append({
                            'text': node.node.text,
                            'score': float(node.score),
                            'metadata': node.node.metadata,
                            'doc_id': node.node.doc_id
                        })
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []

    def get_document_metadata(self) -> List[Dict]:
        """Get metadata about all indexed documents"""
        if not self.index:
            return []
            
        try:
            docstore = self.index.docstore
            return [
                {
                    'id': doc_id,
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'type': doc.metadata.get('file_type', 'Unknown'),
                    'date_added': doc.metadata.get('date_added', 'Unknown'),
                    'categories': doc.metadata.get('categories', []),
                    'summary': doc.metadata.get('summary', '')
                }
                for doc_id, doc in docstore.docs.items()
            ]
        except Exception as e:
            self.logger.error(f"Error getting document metadata: {str(e)}")
            return []

    def add_document(self, 
                    content: Union[str, Document], 
                    metadata: Optional[Dict] = None) -> bool:
        """
        Add a new document to the index
        
        Args:
            content: Document content or Document object
            metadata: Optional metadata if content is a string
            
        Returns:
            bool: Success status
        """
        try:
            if not self.index:
                self._initialize_index()
                
            # Convert string content to Document if needed
            if isinstance(content, str):
                document = Document(
                    text=content,
                    metadata=metadata or {}
                )
            else:
                document = content
            
            # Create TextNode with metadata
            node = TextNode(
                text=document.text,
                metadata=document.metadata
            )
            
            # Insert into index
            self.index.insert_nodes([node])
            self.logger.info(f"Successfully added document with metadata: {document.metadata}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return False

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific document by ID
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data if found, None otherwise
        """
        try:
            if not self.index:
                return None
                
            doc = self.index.docstore.docs.get(doc_id)
            if not doc:
                return None
                
            return {
                'id': doc_id,
                'text': doc.text,
                'metadata': doc.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None

    def update_document_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        Update metadata for a specific document
        
        Args:
            doc_id: Document identifier
            metadata: New metadata to merge with existing
            
        Returns:
            bool: Success status
        """
        try:
            if not self.index:
                return False
                
            doc = self.index.docstore.docs.get(doc_id)
            if not doc:
                return False
                
            # Merge new metadata with existing
            doc.metadata.update(metadata)
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document metadata: {str(e)}")
            return False

    def analyze_document(self, content: Union[bytes, str]) -> Dict:
        """
        Analyze a document's content
        
        Args:
            content: Document content as bytes or string
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Convert bytes to string if needed
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            # Create document with metadata
            doc = Document(
                text=content,
                metadata={
                    'date_analyzed': datetime.now().isoformat(),
                    'file_type': 'text'
                }
            )
            
            # Add to index
            self.add_document(doc)
            
            # Perform basic analysis
            word_count = len(content.split())
            
            # Return analysis results
            return {
                'word_count': word_count,
                'date_analyzed': doc.metadata['date_added'],
                'summary': content[:200] + '...' if len(content) > 200 else content,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing document: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def execute(self, query: str) -> str:
        """
        Main execution method for the tool
        
        Args:
            query: Search query
            
        Returns:
            Formatted results string
        """
        results = self.search_documents(query)
        if not results:
            return "No relevant documents found."
            
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
                f"Relevance: {result['score']:.2f}\n"
                f"Content: {result['text']}\n"
            )
            
        return "\n\n".join(formatted_results)

class LegalDocSearchTool(DocumentTool):
    """Tool for searching and analyzing legal documents, inherits from DocumentTool"""
    pass

def main():
    """Main function for testing"""
    # Create the tool
    tool = DocumentTool(docs_path='legal_docs')
    
    # Add a test document
    test_doc = Document(
        text='This is a sample legal document discussing civil rights.',
        metadata={
            'filename': 'civil_rights_case.txt',
            'categories': ['civil_rights'],
            'date_added': datetime.now().isoformat()
        }
    )
    tool.add_document(test_doc)
    
    # Test search
    results = tool.execute('civil rights')
    print(results)

if __name__ == '__main__':
    main()
