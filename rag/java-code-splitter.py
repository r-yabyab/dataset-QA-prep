#!/usr/bin/env python3
"""

"""

import os
import json
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# For LangChain - install with: pip install langchain-text-splitters
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# For vector database
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_text_splitters import CharacterTextSplitter
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Java separators for reference but use the most appropriate one for CharacterTextSplitter
java_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
# For CharacterTextSplitter, we need to use a single separator
# Using '\n\n' (double newline) as the primary separator for Java code
java_primary_separator = '\n\n'


class JavaCodeSplitter:
    """Java code splitter using LangChain's text splitter with vector database support"""
    
    def __init__(self, java_data_path: str, chunk_size: int = 400, chunk_overlap: int = 50, 
                 vector_db_path: str = "./vectordb", create_vector_db: bool = True):
        self.java_data_path = Path(java_data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_chunks: List[Dict] = []
        self.create_vector_db = create_vector_db
        self.documents: List[Document] = []

        # Use CharacterTextSplitter with a single separator (double newline for Java)
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator=java_primary_separator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Store Java separators for reference
        self.java_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
        self.primary_separator = java_primary_separator
        
        # Vector database setup
        if self.create_vector_db:
            self.vector_db_path = vector_db_path
            self.embed_model = HuggingFaceEmbedding()
            self.db = chromadb.PersistentClient(path=self.vector_db_path)
            self.collection_name = "java_langchain_splitter1111"
            self.chroma_collection = self.db.get_or_create_collection(name=self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Configure LlamaIndex settings
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
        
        # Only process .java files
        self.java_extensions = {'.java'}
        
        # Exclude patterns for directories/files to skip
        self.exclude_patterns = {
            '.git', '.gradle', 'target', 'build', '.idea', '.vscode',
            '__pycache__', '.DS_Store', 'node_modules', 'bin', 'out',
            '.settings', '.classpath', '.project'
        }
    
    def should_exclude_path(self, path: Path) -> bool:
        """Check if a path should be excluded based on exclude patterns"""
        path_parts = set(path.parts)
        return bool(path_parts.intersection(self.exclude_patterns))
    
    def extract_java_metadata(self, content: str, file_path: Path) -> Dict:
        """Extract basic metadata from Java file content"""
        metadata = {
            'package': None,
            'imports': [],
            'classes': [],
            'interfaces': [],
            'enums': []
        }
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract package
            if line.startswith('package ') and line.endswith(';'):
                metadata['package'] = line[8:-1].strip()
            
            # Extract imports
            elif line.startswith('import ') and line.endswith(';'):
                import_name = line[7:-1].strip()
                if not import_name.startswith('static '):
                    metadata['imports'].append(import_name)
            
            # Extract class names (simplified)
            elif 'class ' in line and (line.startswith('public ') or line.startswith('class ')):
                words = line.split()
                if 'class' in words:
                    class_idx = words.index('class')
                    if class_idx + 1 < len(words):
                        class_name = words[class_idx + 1].split('{')[0].split('<')[0]
                        metadata['classes'].append(class_name)
            
            # Extract interface names
            elif 'interface ' in line and (line.startswith('public ') or line.startswith('interface ')):
                words = line.split()
                if 'interface' in words:
                    interface_idx = words.index('interface')
                    if interface_idx + 1 < len(words):
                        interface_name = words[interface_idx + 1].split('{')[0].split('<')[0]
                        metadata['interfaces'].append(interface_name)
            
            # Extract enum names
            elif 'enum ' in line and (line.startswith('public ') or line.startswith('enum ')):
                words = line.split()
                if 'enum' in words:
                    enum_idx = words.index('enum')
                    if enum_idx + 1 < len(words):
                        enum_name = words[enum_idx + 1].split('{')[0]
                        metadata['enums'].append(enum_name)
        
        return metadata
    
    def process_java_file(self, file_path: Path) -> List[Dict]:
        """Process a single Java file and split it into chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return chunks
        
        # Skip empty files
        if not content.strip():
            return chunks
        
        # Extract metadata from the file
        file_metadata = self.extract_java_metadata(content, file_path)
        
        # Split the Java code into chunks using LangChain
        text_chunks = self.text_splitter.split_text(content)
        
        # Get relative path for cleaner storage
        try:
            relative_path = file_path.relative_to(self.java_data_path)
        except ValueError:
            relative_path = file_path
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            
            chunk_data = {
                'id': f"{relative_path}_{i}",
                'source_file': str(relative_path),
                'absolute_path': str(file_path),
                'chunk_index': i,
                'content': chunk_text,
                'character_count': len(chunk_text),
                'line_count': len(chunk_text.split('\n')),
                'file_metadata': file_metadata,
                'timestamp': datetime.now().isoformat()
            }
            chunks.append(chunk_data)
            
            # Create Document for vector database if enabled
            if self.create_vector_db:
                doc = Document(
                    text=chunk_text,
                    metadata={
                        'source_file': str(relative_path),
                        'chunk_index': i,
                        'file_type': 'java',
                        'package': file_metadata.get('package') or '',
                        'classes': ', '.join(file_metadata.get('classes', [])),  # Convert list to string
                        'interfaces': ', '.join(file_metadata.get('interfaces', []))  # Convert list to string
                    }
                )
                self.documents.append(doc)
        
        return chunks
    
    def process_project(self, project_name: str) -> List[Dict]:
        """Process all Java files in a specific project"""
        project_path = self.java_data_path / project_name
        
        if not project_path.exists() or not project_path.is_dir():
            logger.warning(f"Project path does not exist: {project_path}")
            return []
        
        logger.info(f"Processing Java files in project: {project_name}")
        
        java_file_count = 0
        project_chunks = []
        
        # Recursively find all .java files
        for file_path in project_path.rglob("*.java"):
            if not file_path.is_file() or self.should_exclude_path(file_path):
                continue
            
            file_chunks = self.process_java_file(file_path)
            project_chunks.extend(file_chunks)
            java_file_count += 1
            
            if java_file_count % 50 == 0:
                logger.info(f"Processed {java_file_count} Java files from {project_name}")
        
        logger.info(f"Completed processing {project_name}: {java_file_count} Java files, {len(project_chunks)} chunks")
        return project_chunks
    
    def process_all_projects(self) -> None:
        """Process only the LeetCode-in-Java project"""
        logger.info(f"Starting processing of java-data directory: {self.java_data_path}")
        
        if not self.java_data_path.exists():
            logger.error(f"Java data path does not exist: {self.java_data_path}")
            return
        
        # Only process LeetCode-in-Java project
        target_project = "LeetCode-in-Java"
        target_path = self.java_data_path / target_project
        
        if not target_path.exists() or not target_path.is_dir():
            logger.error(f"Target project not found: {target_path}")
            return
        
        logger.info(f"Processing only: {target_project}")
        
        project_chunks = self.process_project(target_project)
        self.processed_chunks.extend(project_chunks)
        
        logger.info(f"Total Java chunks created: {len(self.processed_chunks)}")
    
    def build_vector_index(self) -> VectorStoreIndex:
        """Build the vector index from all processed documents"""
        if not self.create_vector_db:
            logger.info("Vector database creation is disabled")
            return None
            
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
        
        logger.info(f"Vector index built successfully with {len(self.documents)} documents")
        logger.info(f"Vector database saved to: {self.vector_db_path}")
        logger.info(f"Collection name: {self.collection_name}")
        return index
    
    def save_to_file(self, output_path: str) -> None:
        """Save processed chunks to JSON file"""
        output_file = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare summary data
        summary = {
            'metadata': {
                'total_chunks': len(self.processed_chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'processing_date': datetime.now().isoformat(),
                'source_directory': str(self.java_data_path)
            },
            'chunks': self.processed_chunks
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.processed_chunks)} chunks to {output_file}")
    
    def print_statistics(self) -> None:
        """Print detailed processing statistics"""
        if not self.processed_chunks:
            logger.warning("No chunks to analyze")
            return
        
        logger.info("=" * 60)
        logger.info("JAVA CODE PROCESSING STATISTICS")
        logger.info("=" * 60)
        
        # LangChain Java splitter info
        logger.info("LangChain Text Splitter Configuration:")
        logger.info(f"  Language: Java (Language.JAVA)")
        logger.info(f"  Java separators used: {self.java_separators}")
        logger.info("")
        
        # Basic stats
        total_chunks = len(self.processed_chunks)
        logger.info(f"Total Java chunks: {total_chunks}")
        logger.info(f"Chunk size setting: {self.chunk_size} tokens")
        logger.info(f"Chunk overlap setting: {self.chunk_overlap} tokens")
        
        # Character and line statistics
        char_counts = [chunk['character_count'] for chunk in self.processed_chunks]
        line_counts = [chunk['line_count'] for chunk in self.processed_chunks]
        
        logger.info(f"Average chunk size: {sum(char_counts) / len(char_counts):.1f} characters")
        logger.info(f"Min chunk size: {min(char_counts)} characters")
        logger.info(f"Max chunk size: {max(char_counts)} characters")
        logger.info(f"Average lines per chunk: {sum(line_counts) / len(line_counts):.1f}")
        
        # File distribution
        files = set(chunk['source_file'] for chunk in self.processed_chunks)
        logger.info(f"Total Java files processed: {len(files)}")
        logger.info(f"Average chunks per file: {total_chunks / len(files):.1f}")
        
        # Project distribution
        projects = set(chunk['source_file'].split('/')[0] if '/' in chunk['source_file'] 
                      else chunk['source_file'].split('\\')[0] for chunk in self.processed_chunks)
        logger.info(f"Projects processed: {len(projects)}")
        
        for project in sorted(projects):
            project_chunks = [c for c in self.processed_chunks 
                            if c['source_file'].startswith(project)]
            logger.info(f"  {project}: {len(project_chunks)} chunks")
        
        # Vector database info
        if self.create_vector_db:
            logger.info("")
            logger.info("Vector Database Information:")
            logger.info(f"  Total documents in vector DB: {len(self.documents)}")
            logger.info(f"  Vector DB path: {self.vector_db_path}")
            logger.info(f"  Collection name: {self.collection_name}")
            logger.info(f"  Embedding model: HuggingFace (sentence-transformers)")
        else:
            logger.info("")
            logger.info("Vector database creation: DISABLED")


def main():
    """Main function to process Java files using LangChain's Language.JAVA splitter"""
    # Configuration
    java_data_path = r"w:\Users\cayab\dataset-QA-prep\data\java-data"
    output_path = r"w:\Users\cayab\dataset-QA-prep\rag\outputs\java_chunks_langchain.json"
    vector_db_path = r"w:\Users\cayab\dataset-QA-prep\rag\vectordb"
    chunk_size = 500  # tokens
    chunk_overlap = 50  # tokens
    create_vector_db = True  # Set to False if you only want JSON output
    
    logger.info("Starting Java code processing with LangChain Language.JAVA splitter")
    logger.info(f"Using RecursiveCharacterTextSplitter.from_language(Language.JAVA)")
    logger.info(f"Source directory: {java_data_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Vector DB path: {vector_db_path}")
    logger.info(f"Create vector database: {create_vector_db}")
    logger.info(f"Chunk size: {chunk_size} tokens")
    logger.info(f"Chunk overlap: {chunk_overlap} tokens")
    
    # Show Java separators that will be used
    # java_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
    logger.info(f"Java-specific separators: {java_separators}")
    logger.info("")
    
    # Initialize processor
    processor = JavaCodeSplitter(
        java_data_path=java_data_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_db_path=vector_db_path,
        create_vector_db=create_vector_db
    )
    
    # Process all projects
    processor.process_all_projects()
    
    if processor.processed_chunks:
        # Save JSON results
        processor.save_to_file(output_path)
        
        # Build vector database if enabled
        if create_vector_db:
            vector_index = processor.build_vector_index()
            if vector_index:
                logger.info("Vector database created successfully!")
            else:
                logger.error("Failed to create vector database")
        
        # Print statistics
        processor.print_statistics()
        
        logger.info("Java code processing complete!")
        logger.info(f"JSON results saved to: {output_path}")
        if create_vector_db:
            logger.info(f"Vector database saved to: {vector_db_path}")
    else:
        logger.error("No Java chunks were processed")


if __name__ == "__main__":
    main()
