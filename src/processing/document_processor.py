"""Document processing and content extraction"""
import os
import glob
import logging
import base64
import io
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available - .docx files will be skipped")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF processing will be limited")


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
    
    def process_base64_document(self, base64_content: str, file_name: str, file_extension: str) -> Optional[str]:
        """
        Process a document from base64-encoded content (supports PDF and DOCX)
        
        Args:
            base64_content: Base64-encoded document content
            file_name: Original filename for logging
            file_extension: File extension (pdf, docx, doc)
            
        Returns:
            Extracted text content or None on error
        """
        file_ext = file_extension.lower().strip('.')
        
        try:
            # Decode base64 to binary
            file_bytes = base64.b64decode(base64_content)
            
            # Route to appropriate processor
            if file_ext == 'pdf':
                result = self._process_pdf_bytes(file_bytes, file_name)
            elif file_ext in ['docx', 'doc']:
                result = self._process_docx_bytes(file_bytes, file_name)
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return None
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to process document {file_name}: {e}")
            return None
    
    def _process_pdf_bytes(self, file_bytes: bytes, file_name: str) -> Optional[str]:
        """Extract text from PDF bytes"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not available")
            return None
        
        try:
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            text_content = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_content.append(text.strip())
            
            full_text = '\n\n'.join(text_content)
            logger.info(f"PDF processed: {file_name} ({num_pages} pages, {len(full_text)} chars)")
            
            return full_text
        
        except Exception as e:
            logger.error(f"Failed to extract PDF {file_name}: {e}")
            return None
    
    def _process_docx_bytes(self, file_bytes: bytes, file_name: str) -> Optional[str]:
        """Extract text from DOCX bytes"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available")
            return None
        
        try:
            docx_file = io.BytesIO(file_bytes)
            doc = Document(docx_file)
            
            # Extract text from paragraphs
            paragraphs = []
            for p in enumerate(doc.paragraphs, 1):
                if p.text.strip():
                    paragraphs.append(p.text.strip())
            
            # Extract text from tables if any
            if doc.tables:
                for table in doc.tables:
                    for row in table.rows:
                        row_text = ' | '.join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text:
                            paragraphs.append(row_text)
            
            full_text = '\n\n'.join(paragraphs)
            logger.info(f"DOCX processed: {file_name} ({len(paragraphs)} paragraphs, {len(full_text)} chars)")
            
            return full_text
        
        except Exception as e:
            logger.error(f"Failed to extract DOCX {file_name}: {e}")
            return None
