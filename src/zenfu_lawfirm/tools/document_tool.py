# src/zenfu_lawfirm/tools/document_tool.py
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document as LlamaDocument
from llama_index.data_structs.node import Node
from llama_index.errors import DOC_NOT_FOUND
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
            extra_info=self.metadata
        )

class LegalDocSearchTool:
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
                documents = SimpleDirectoryReader(self.docs_path).load_data()
                self.index = GPTVectorStoreIndex.from_documents(documents)
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
                
            results = self.index.query(
                query,
                top_k=top_k,
                include_metadata=True
            )
            
            filtered_results = []
            for result in results:
                if result.score >= min_score:
                    filtered_results.append({
                        'text': str(result.text),
                        'score': float(result.score),
                        'metadata': result.metadata,
                        'doc_id': result.doc_id
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
            return [
                {
                    'id': doc.doc_id,
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'type': doc.metadata.get('file_type', 'Unknown'),
                    'date_added': doc.metadata.get('date_added', 'Unknown'),
                    'categories': doc.metadata.get('categories', []),
                    'summary': doc.metadata.get('summary', '')
                }
                for doc in self.index.docstore.docs.values()
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
            
            # Convert to LlamaIndex Node
            node = Node(
                text=document.text,
                extra_info=document.metadata
            )
            
            self.index.insert(node)
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
                'id': doc.doc_id,
                'text': doc.text,
                'metadata': doc.extra_info
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
            doc.extra_info.update(metadata)
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document metadata: {str(e)}")
            return False

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

def main():
    """Main function for testing"""
    # Create the tool
    tool = LegalDocSearchTool(docs_path='legal_docs')
    
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