#!/usr/bin/env python3
"""
Build Vector Database for Java Files Only

This script processes only Java files (.java) from Java projects and creates a vector database 
using LangChain's code splitter specifically designed for Java code.

Features:
- Recursively scans for .java files only
- Uses LangChain's RecursiveCharacterTextSplitter with Java-specific separators
- Creates embeddings for retrieval
- Uses ChromaDB vector store
"""

import os
from pathlib import Path
from typing import List
import logging

from config import VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JavaFileProcessor:
    """Processor specifically for Java files using LangChain's code splitter"""
    
    def __init__(self, java_data_path: str):
        self.java_data_path = Path(java_data_path)
        self.documents: List[Document] = []
        
        # Java-specific code splitter with appropriate separators
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                # Java class and method separators
                "\n\n",           # Double newlines (natural breaks)
                "\npublic class ", # Class declarations
                "\nprivate class ",
                "\nprotected class ",
                "\nclass ",
                "\npublic interface ",
                "\ninterface ",
                "\npublic ",       # Method declarations
                "\nprivate ",
                "\nprotected ",
                "\nstatic ",
                "\n@",            # Annotations
                "\n//",           # Single line comments
                "\n/*",           # Multi-line comments start
                "\n*/",           # Multi-line comments end
                "\nimport ",      # Import statements
                "\npackage ",     # Package statements
                "\n{",            # Opening braces
                "\n}",            # Closing braces
                "\n;",            # Statement endings
                "\n",             # Single newlines
                " {",             # Opening braces with space
                " }",             # Closing braces with space
                "; ",             # Statement endings with space
                ", ",             # Commas
                " ",              # Spaces
                ""                # Character level
            ]
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding()
        
        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        # Use a specific collection name for Java-only files
        self.collection_name = "java_files_only"
        self.chroma_collection = self.db.get_or_create_collection(name=self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Configure LlamaIndex settings
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        
        # Only process .java files
        self.java_extensions = {'.java'}
        
        # Exclude patterns for directories/files to skip
        self.exclude_patterns = {
            '.git', '.gradle', 'target', 'build', '.idea', '.vscode',
            '__pycache__', '.DS_Store', 'node_modules', 'bin', 'out'
        }
    
    def should_exclude_path(self, path: Path) -> bool:
        """Check if a path should be excluded based on exclude patterns"""
        path_parts = set(path.parts)
        return bool(path_parts.intersection(self.exclude_patterns))
    
    def parse_java_file(self, file_path: Path) -> List[Document]:
        """Parse a single Java file and split it into chunks"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return documents
        
        # Skip empty files
        if not content.strip():
            return documents
        
        # Split the Java code into chunks
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            
            # Create document with minimal metadata for debugging
            doc = Document(
                text=chunk_text,
                metadata={
                    'source_file': str(file_path),
                    'chunk_index': i,
                    'file_type': 'java'
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_project(self, project_name: str) -> None:
        """Process all Java files in a specific project"""
        project_path = self.java_data_path / project_name
        
        if not project_path.exists() or not project_path.is_dir():
            logger.warning(f"Project path does not exist: {project_path}")
            return
        
        logger.info(f"Processing Java files in project: {project_name}")
        
        java_file_count = 0
        project_documents = []
        
        # Recursively find all .java files
        for file_path in project_path.rglob("*.java"):
            if not file_path.is_file() or self.should_exclude_path(file_path):
                continue
            
            documents = self.parse_java_file(file_path)
            project_documents.extend(documents)
            java_file_count += 1
            
            if java_file_count % 50 == 0:
                logger.info(f"Processed {java_file_count} Java files from {project_name}")
        
        self.documents.extend(project_documents)
        logger.info(f"Completed processing {project_name}: {java_file_count} Java files, {len(project_documents)} document chunks")
    
    def build_vector_index(self) -> VectorStoreIndex:
        """Build the vector index from all processed documents"""
        logger.info(f"Building vector index with {len(self.documents)} document chunks")
        
        if not self.documents:
            logger.warning("No documents to index!")
            return None
        
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        logger.info("Vector index built successfully")
        return index
    
    def process_all_projects(self) -> VectorStoreIndex:
        """Process all configured projects"""
        logger.info(f"Starting processing of java-data directory: {self.java_data_path}")
        
        # Get all subdirectories in java-data as projects
        if not self.java_data_path.exists():
            logger.error(f"Java data path does not exist: {self.java_data_path}")
            return None
        
        projects = [d for d in self.java_data_path.iterdir() if d.is_dir()]
        
        if not projects:
            logger.warning(f"No projects found in {self.java_data_path}")
            return None
        
        logger.info(f"Found {len(projects)} projects to process")
        
        for project_dir in projects:
            self.process_project(project_dir.name)
        
        logger.info(f"Total Java document chunks created: {len(self.documents)}")
        return self.build_vector_index()
    
    def print_statistics(self) -> None:
        """Print processing statistics"""
        logger.info("=" * 50)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total Java document chunks: {len(self.documents)}")
        logger.info(f"Chunk size: {CHUNK_SIZE} tokens")
        logger.info(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
        logger.info(f"Collection name: {self.collection_name}")
        logger.info(f"Vector database path: {VECTOR_DB_PATH}")
        
        if self.documents:
            # Sample chunk statistics
            chunk_lengths = [len(doc.text) for doc in self.documents]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.info(f"Average chunk length: {avg_length:.1f} characters")
            logger.info(f"Min chunk length: {min(chunk_lengths)} characters")
            logger.info(f"Max chunk length: {max(chunk_lengths)} characters")


def main():
    """Main function to process Java files and build vector database"""
    java_data_path = r"w:\Users\cayab\dataset-QA-prep\data\java-data"
    
    logger.info("Starting Java-only vector database creation")
    logger.info(f"Source directory: {java_data_path}")
    
    processor = JavaFileProcessor(java_data_path)
    index = processor.process_all_projects()
    
    if index:
        processor.print_statistics()
        logger.info("Vector database preparation complete!")
    else:
        logger.error("Failed to create vector database")
    
    return index


if __name__ == "__main__":
    main()
