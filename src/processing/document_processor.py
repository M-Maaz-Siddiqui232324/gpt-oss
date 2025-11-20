"""Document processing and content extraction"""
import os
import glob
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Optional imports
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available - .docx files will be skipped")


class DocumentProcessor:
    """Handles document loading and content extraction"""
    
    def __init__(self, docs_folder: str):
        self.docs_folder = docs_folder
        logger.info(f"Initialized DocumentProcessor with folder: {docs_folder}")
    
    def load_documents(self) -> List[Dict]:
        """Load all documents from the docs folder"""
        logger.info(f"Starting document loading from: {self.docs_folder}")
        
        if not os.path.exists(self.docs_folder):
            logger.error(f"Docs folder not found: {self.docs_folder}")
            return []
        
        documents = []
        file_pattern = os.path.join(self.docs_folder, "**/*")
        
        for file_path in glob.glob(file_pattern, recursive=True):
            if os.path.isfile(file_path):
                logger.debug(f"Processing file: {file_path}")
                content = self._extract_content(file_path)
                
                if content and content.strip():
                    documents.append({
                        'path': file_path,
                        'name': os.path.basename(file_path),
                        'content': content,
                        'type': os.path.splitext(file_path)[1][1:] or 'unknown'
                    })
                    logger.info(f"Loaded: {os.path.basename(file_path)} ({len(content)} chars)")
                else:
                    logger.warning(f"Skipped empty/unreadable: {file_path}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _extract_content(self, file_path: str) -> str:
        """Extract text content from DOCX files only"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Word documents only
            if file_ext == '.docx' and DOCX_AVAILABLE:
                logger.debug(f"Extracting DOCX: {file_path}")
                doc = Document(file_path)
                return '\n\n'.join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            else:
                logger.warning(f"Unsupported file type: {file_path} (only .docx supported)")
                return ""
        
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {str(e)}")
            return ""
