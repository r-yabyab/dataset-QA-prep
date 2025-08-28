#!/usr/bin/env python3
"""
Build Vector Database for Java Projects RAG System

This script processes Java projects (netty, Telegram-X, termux-app) and creates
a vector database with appropriate metadata for RAG (Retrieval Augmented Generation).

Features:
- Recursively scans Java projects
- Extracts metadata (project, module, package, class info)
- Splits text using LangChain with 500 token chunks
- Creates embeddings with rich metadata for better retrieval
- Uses ChromaDB vector store compatible with simple-vector.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import VECTOR_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
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

class JavaProjectProcessor:
    """Main processor for Java projects in the java-data directory"""
    
    def __init__(self, java_data_path: str):
        self.java_data_path = Path(java_data_path)
        self.documents: List[Document] = []
        
        # Initialize text splitter with token-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                "\n\n    ",  # Indented blocks (common in code)
                "\n    ",    # Single-level indentation
                "\n\t",     # Tab indentation
                ". ",    # Sentence endings
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character breaks (last resort)
            ]
        )
        
        # Initialize embedding model and vector store
        self.embed_model = HuggingFaceEmbedding()
        
        # Set up persistent ChromaDB vector store
        self.db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.chroma_collection = self.db.get_or_create_collection(name=COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        
        self.java_extensions = {'.java'}
        self.doc_extensions = {'.md', '.txt', '.rst', '.adoc'}
        self.config_extensions = {'.xml', '.gradle', '.kts', '.properties', '.yml', '.yaml'}
        
        self.exclude_patterns = {
            '.git', '.gradle', 'target', 'build', '.idea', '.vscode',
            '__pycache__', '.DS_Store', 'node_modules'
        }
        
        self.project_configs = {
            'netty': {'type': 'maven', 'language': 'java'},
            'Telegram-X': {'type': 'gradle', 'language': 'java'},
            'termux-app': {'type': 'gradle', 'language': 'java'},
            'LeetCode-in-Java': {'type': 'maven', 'language': 'java'}
        }
    
    def should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded based on exclude patterns"""
        path_parts = set(path.parts)
        return bool(path_parts.intersection(self.exclude_patterns))
    
    def extract_java_metadata(self, file_path: Path, project_name: str) -> Dict[str, Any]:
        """Extract metadata from Java file path and content"""
        relative_path = file_path.relative_to(self.java_data_path / project_name)
        
        parts = list(relative_path.parts)
        package = None
        module = None
        source_type = "main"
        
        if "src" in parts:
            src_idx = parts.index("src")
            if src_idx + 1 < len(parts):
                source_type = parts[src_idx + 1]  # main, test, etc.
            
            if src_idx + 3 < len(parts) and parts[src_idx + 2] == "java":
                package_parts = parts[src_idx + 3:-1]  # Exclude filename
                package = ".".join(package_parts) if package_parts else None
        
        if len(parts) > 1:
            module = parts[0]
        
        class_name = file_path.stem
        
        return {
            "project_name": project_name,
            "file_type": "java_source",
            "relative_path": str(relative_path),
            "package": package,
            "class_name": class_name,
            "module": module,
            "source_type": source_type,
            "file_extension": file_path.suffix
        }
    
    def extract_doc_metadata(self, file_path: Path, project_name: str) -> Dict[str, Any]:
        """Extract metadata from documentation files"""
        relative_path = file_path.relative_to(self.java_data_path / project_name)
        
        filename = file_path.name.lower()
        doc_type = "general"
        
        if "readme" in filename:
            doc_type = "readme"
        elif "contributing" in filename:
            doc_type = "contributing"
        elif "changelog" in filename or "changes" in filename:
            doc_type = "changelog"
        elif "license" in filename:
            doc_type = "license"
        elif "security" in filename:
            doc_type = "security"
        elif "guide" in filename:
            doc_type = "guide"
        elif "install" in filename:
            doc_type = "installation"
        
        parts = list(relative_path.parts)
        scope = "project_root" if len(parts) == 1 else f"module_{parts[0]}"
        
        return {
            "project_name": project_name,
            "file_type": "documentation",
            "relative_path": str(relative_path),
            "doc_type": doc_type,
            "scope": scope,
            "file_extension": file_path.suffix
        }
    
    def extract_config_metadata(self, file_path: Path, project_name: str) -> Dict[str, Any]:
        """Extract metadata from configuration files"""
        relative_path = file_path.relative_to(self.java_data_path / project_name)
        
        filename = file_path.name.lower()
        config_type = "general"
        
        if "pom.xml" in filename:
            config_type = "maven_pom"
        elif "build.gradle" in filename:
            config_type = "gradle_build"
        elif "settings.gradle" in filename:
            config_type = "gradle_settings"
        elif ".properties" in filename:
            config_type = "properties"
        elif filename.endswith(('.yml', '.yaml')):
            config_type = "yaml_config"
        
        return {
            "project_name": project_name,
            "file_type": "configuration",
            "relative_path": str(relative_path),
            "config_type": config_type,
            "file_extension": file_path.suffix
        }
    
    def parse_java_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Document]:
        """Parse Java file and split into logical chunks using LangChain"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return documents
        
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['chunk_type'] = 'code_chunk'
            chunk_metadata['total_chunks'] = len(text_chunks)
            
            doc = Document(
                text=chunk_text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def parse_markdown_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Document]:
        """Parse Markdown file and split using LangChain"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return documents
        
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            lines = chunk_text.split('\n')
            title = None
            for line in lines:
                if line.strip().startswith('#'):
                    title = line.strip('#').strip()
                    break
            
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['chunk_type'] = 'documentation_chunk'
            chunk_metadata['total_chunks'] = len(text_chunks)
            if title:
                chunk_metadata['section_title'] = title
            
            doc = Document(
                text=chunk_text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def parse_text_file(self, file_path: Path, metadata: Dict[str, Any]) -> List[Document]:
        """Parse text/config files using LangChain"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return documents
        
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['chunk_type'] = 'config_chunk'
            chunk_metadata['total_chunks'] = len(text_chunks)
            
            doc = Document(
                text=chunk_text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_project(self, project_name: str) -> None:
        """Process a single project directory"""
        project_path = self.java_data_path / project_name
        
        if not project_path.exists() or not project_path.is_dir():
            logger.warning(f"Project path does not exist: {project_path}")
            return
        
        logger.info(f"Processing project: {project_name}")
        
        file_count = 0
        project_documents = []
        
        for file_path in project_path.rglob("*"):
            if not file_path.is_file() or self.should_exclude_path(file_path):
                continue
            
            file_ext = file_path.suffix.lower()
            documents = []
            
            try:
                if file_ext in self.java_extensions:
                    metadata = self.extract_java_metadata(file_path, project_name)
                    documents = self.parse_java_file(file_path, metadata)
                
                elif file_ext in self.doc_extensions:
                    metadata = self.extract_doc_metadata(file_path, project_name)
                    if file_ext == '.md':
                        documents = self.parse_markdown_file(file_path, metadata)
                    else:
                        documents = self.parse_text_file(file_path, metadata)
                
                elif file_ext in self.config_extensions:
                    metadata = self.extract_config_metadata(file_path, project_name)
                    documents = self.parse_text_file(file_path, metadata)
                
                project_documents.extend(documents)
                file_count += 1
                
                if file_count % 100 == 0:
                    logger.info(f"Processed {file_count} files from {project_name}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        self.documents.extend(project_documents)
        logger.info(f"Completed processing {project_name}: {file_count} files, {len(project_documents)} document chunks")
    
    def build_vector_index(self) -> VectorStoreIndex:
        """Build and return the vector index"""
        logger.info(f"Building vector index with {len(self.documents)} documents")
        
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
        """Process all projects in the java-data directory and build vector index"""
        logger.info(f"Starting processing of java-data directory: {self.java_data_path}")
        
        for project_name in self.project_configs.keys():
            self.process_project(project_name)
        
        logger.info(f"Total documents created: {len(self.documents)}")
        
        return self.build_vector_index()
    
    def print_statistics(self) -> None:
        """Print processing statistics"""
        if not self.documents:
            logger.info("No documents to analyze")
            return
        
        project_stats = {}
        file_type_stats = {}
        
        for doc in self.documents:
            metadata = doc.metadata
            project = metadata.get('project_name', 'unknown')
            file_type = metadata.get('file_type', 'unknown')
            
            project_stats[project] = project_stats.get(project, 0) + 1
            file_type_stats[file_type] = file_type_stats.get(file_type, 0) + 1
        
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total document chunks: {len(self.documents)}")
        logger.info(f"Chunk size: {CHUNK_SIZE} tokens")
        logger.info(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
        logger.info("\nBy Project:")
        for project, count in sorted(project_stats.items()):
            logger.info(f"  {project}: {count}")
        
        logger.info("\nBy File Type:")
        for file_type, count in sorted(file_type_stats.items()):
            logger.info(f"  {file_type}: {count}")

def main():
    """Main entry point"""
    java_data_path = r"w:\Users\cayab\dataset-QA-prep\data\java-data"
    
    processor = JavaProjectProcessor(java_data_path)
    
    index = processor.process_all_projects()
    
    processor.print_statistics()
    
    logger.info(f"Vector database created at: {VECTOR_DB_PATH}")
    logger.info(f"Collection name: {COLLECTION_NAME}")
    logger.info("Vector database preparation complete!")
    
    return index

if __name__ == "__main__":
    main()
