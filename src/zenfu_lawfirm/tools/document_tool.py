# src/zenfu_lawfirm/tools/document_tool.py

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.data_structs.node import Node
from llama_index.errors import DOC_NOT_FOUND
from typing import List, Dict, Optional
import os
import logging

# Set up logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class Document:
    """Represents a document with its text and metadata"""
    
    def __init__(self, text: str, metadata: Dict = None):
        self.text = text
        self.metadata = metadata or {}

class LegalDocSearchTool:
    """Tool for searching and analyzing legal documents"""
    
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the document index"""
        try:
            if os.path.exists(self.docs_path):
                documents = SimpleDirectoryReader(self.docs_path).load_data()
                self.index = GPTVectorStoreIndex.from_documents(documents)
                logging.info(f"Initialized document index with {len(documents)} documents")
            else:
                self.index = None
                logging.warning(f"Document path {self.docs_path} does not exist")
        except Exception as e:
            logging.error(f"Error initializing document index: {str(e)}")
            self.index = None

    def search_documents(self, 
                        query: str, 
                        top_k: int = 5) -> List[Dict]:
        """
        Search documents for relevant information
        
        Args:
            query: Search query
            top_k: Number of results to return
            
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
            
            return [
                {
                    'text': str(result.text),
                    'score': float(result.score),
                    'metadata': result.metadata
                }
                for result in results
            ]
            
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}")
            return []

    def get_document_metadata(self) -> List[Dict]:
        """Get metadata about all indexed documents"""
        if not self.index:
            return []
            
        return [
            {
                'id': doc.doc_id,
                'filename': doc.metadata.get('filename', 'Unknown'),
                'type': doc.metadata.get('file_type', 'Unknown'),
                'date_added': doc.metadata.get('date_added', 'Unknown')
            }
            for doc in self.index.docstore.docs.values()
        ]

    def add_document(self, 
                    content: str, 
                    metadata: Dict = None) -> bool:
        """Add a new document to the index"""
        try:
            if not self.index:
                self._initialize_index()
                
            document = Document(
                text=content,
                metadata=metadata or {}
            )
            
            # Convert the Document to Node
            node = Node(
                text=document.text,
                id=None,
                extra_info=document.metadata
            )
            
            self.index.insert(node)
            return True
            
        except Exception as e:
            logging.error(f"Error adding document: {str(e)}")
            return False

    def execute(self, query: str) -> str:
        """Main execution method for the tool"""
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
    # Create the tool
    tool = LegalDocSearchTool(docs_path='path_to_your_documents')
    
    # Add a document
    tool.add_document('This is a sample document.', {'filename': 'sample.txt', 'date_added': '2022-01-01'})
    
    # Search documents
    results = tool.execute('sample')
    print(results)

if __name__ == '__main__':
    main()
