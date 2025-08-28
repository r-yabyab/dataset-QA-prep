#!/usr/bin/env python3
"""
Build Vector Database for Java Projects RAG System (No Metadata)

This script processes Java projects and creates a vector database for RAG (Retrieval Augmented Generation) without attaching any metadata to the document chunks.

Features:
- Recursively scans Java projects
- Splits text using LangChain with 500 token chunks
- Creates embeddings for retrieval (no metadata)
- Uses ChromaDB vector store compatible with simple-vector.py
"""

import os
from pathlib import Path
from typing import List
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

class JavaProjectProcessorNoMetadata:
    """Processor for Java projects in the java-data directory (no metadata)"""
    
    def __init__(self, java_data_path: str):
        self.java_data_path = Path(java_data_path)
        self.documents: List[Document] = []
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n", "\n", "\n\n    ", "\n    ", "\n\t", ". ", ", ", " ", ""
            ]
        )
        
        self.embed_model = HuggingFaceEmbedding()
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
            # 'netty': {},
            # 'Telegram-X': {},
            # 'termux-app': {},
            'LeetCode-in-Java': {}
        }
    
    def should_exclude_path(self, path: Path) -> bool:
        path_parts = set(path.parts)
        return bool(path_parts.intersection(self.exclude_patterns))
    
    def parse_file(self, file_path: Path) -> List[Document]:
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return documents
        text_chunks = self.text_splitter.split_text(content)
        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue
            doc = Document(text=chunk_text)
            documents.append(doc)
        return documents
    
    def process_project(self, project_name: str) -> None:
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
            if file_ext in self.java_extensions or file_ext in self.doc_extensions or file_ext in self.config_extensions:
                documents = self.parse_file(file_path)
                project_documents.extend(documents)
                file_count += 1
                if file_count % 100 == 0:
                    logger.info(f"Processed {file_count} files from {project_name}")
        self.documents.extend(project_documents)
        logger.info(f"Completed processing {project_name}: {file_count} files, {len(project_documents)} document chunks")
    
    def build_vector_index(self) -> VectorStoreIndex:
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
        logger.info(f"Starting processing of java-data directory: {self.java_data_path}")
        for project_name in self.project_configs.keys():
            self.process_project(project_name)
        logger.info(f"Total documents created: {len(self.documents)}")
        return self.build_vector_index()
    
    def print_statistics(self) -> None:
        logger.info(f"Total document chunks: {len(self.documents)}")
        logger.info(f"Chunk size: {CHUNK_SIZE} tokens")
        logger.info(f"Chunk overlap: {CHUNK_OVERLAP} tokens")


def main():
    java_data_path = r"w:\Users\cayab\dataset-QA-prep\data\java-data"
    processor = JavaProjectProcessorNoMetadata(java_data_path)
    index = processor.process_all_projects()
    processor.print_statistics()
    logger.info(f"Vector database created at: {VECTOR_DB_PATH}")
    logger.info(f"Collection name: {COLLECTION_NAME}")
    logger.info("Vector database preparation complete!")
    return index

if __name__ == "__main__":
    main()
