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
            logger.info("="*60)
            logger.info("📄 DOCUMENT PROCESSING STARTED")
            logger.info(f"   File name: {file_name}")
            logger.info(f"   File type: {file_ext.upper()}")
            logger.info(f"   Base64 content length: {len(base64_content)} characters")
            
            # Decode base64 to binary
            logger.info("🔓 Decoding base64 content...")
            file_bytes = base64.b64decode(base64_content)
            logger.info(f"✅ Decoded successfully")
            logger.info(f"   File size: {len(file_bytes):,} bytes ({len(file_bytes) / 1024:.2f} KB)")
            
            # Route to appropriate processor
            if file_ext == 'pdf':
                logger.info("📕 Routing to PDF processor...")
                result = self._process_pdf_bytes(file_bytes, file_name)
            elif file_ext in ['docx', 'doc']:
                logger.info("📘 Routing to DOCX processor...")
                result = self._process_docx_bytes(file_bytes, file_name)
            else:
                logger.error(f"❌ Unsupported file type: {file_ext}")
                logger.error(f"   Supported types: pdf, docx, doc")
                return None
            
            if result:
                logger.info("="*60)
                logger.info("✅ DOCUMENT PROCESSING COMPLETED")
                logger.info(f"   Extracted text length: {len(result):,} characters")
                logger.info(f"   Text preview: {result[:200]}...")
                logger.info("="*60)
            else:
                logger.error("❌ Document processing returned empty result")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Failed to process document {file_name}: {e}", exc_info=True)
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            return None
    
    def _process_pdf_bytes(self, file_bytes: bytes, file_name: str) -> Optional[str]:
        """Extract text from PDF bytes"""
        if not PDF_AVAILABLE:
            logger.error("❌ PyPDF2 not available - cannot process PDF")
            logger.error("   Install with: pip install PyPDF2")
            return None
        
        try:
            logger.info("📕 Processing PDF document...")
            logger.info(f"   File: {file_name}")
            logger.info(f"   Size: {len(file_bytes):,} bytes")
            
            # Create file-like object
            logger.info("🔄 Creating PDF reader...")
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            num_pages = len(pdf_reader.pages)
            logger.info(f"✅ PDF loaded successfully")
            logger.info(f"   Total pages: {num_pages}")
            
            # Extract metadata if available
            if pdf_reader.metadata:
                logger.info("📋 PDF Metadata:")
                if pdf_reader.metadata.title:
                    logger.info(f"   Title: {pdf_reader.metadata.title}")
                if pdf_reader.metadata.author:
                    logger.info(f"   Author: {pdf_reader.metadata.author}")
            
            # Extract text from each page
            logger.info("📖 Extracting text from pages...")
            text_content = []
            for page_num in range(num_pages):
                logger.info(f"   Processing page {page_num + 1}/{num_pages}...")
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_content.append(text.strip())
                    logger.info(f"   ✅ Page {page_num + 1}: Extracted {len(text):,} characters")
                else:
                    logger.warning(f"   ⚠️  Page {page_num + 1}: No text extracted (might be image-based)")
            
            full_text = '\n\n'.join(text_content)
            logger.info("="*60)
            logger.info("✅ PDF TEXT EXTRACTION COMPLETED")
            logger.info(f"   Pages processed: {num_pages}")
            logger.info(f"   Pages with text: {len(text_content)}")
            logger.info(f"   Total characters: {len(full_text):,}")
            logger.info(f"   Total words (approx): {len(full_text.split()):,}")
            logger.info("="*60)
            
            return full_text
        
        except Exception as e:
            logger.error(f"❌ Failed to extract text from PDF {file_name}: {e}", exc_info=True)
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            return None
    
    def _process_docx_bytes(self, file_bytes: bytes, file_name: str) -> Optional[str]:
        """Extract text from DOCX bytes"""
        if not DOCX_AVAILABLE:
            logger.error("❌ python-docx not available - cannot process DOCX")
            logger.error("   Install with: pip install python-docx")
            return None
        
        try:
            logger.info("📘 Processing DOCX document...")
            logger.info(f"   File: {file_name}")
            logger.info(f"   Size: {len(file_bytes):,} bytes")
            
            # Create file-like object
            logger.info("🔄 Creating DOCX reader...")
            docx_file = io.BytesIO(file_bytes)
            doc = Document(docx_file)
            logger.info("✅ DOCX loaded successfully")
            
            # Extract text from paragraphs
            logger.info("📖 Extracting text from paragraphs...")
            paragraphs = []
            total_paragraphs = len(doc.paragraphs)
            logger.info(f"   Total paragraphs in document: {total_paragraphs}")
            
            for i, p in enumerate(doc.paragraphs, 1):
                if p.text.strip():
                    paragraphs.append(p.text.strip())
                    if i <= 3:  # Log first 3 paragraphs
                        logger.info(f"   Paragraph {i}: {len(p.text)} chars - {p.text[:100]}...")
            
            logger.info(f"   Non-empty paragraphs: {len(paragraphs)}")
            
            # Extract text from tables if any
            if doc.tables:
                logger.info(f"📊 Document contains {len(doc.tables)} table(s)")
                logger.info("   Extracting text from tables...")
                for table_num, table in enumerate(doc.tables, 1):
                    for row in table.rows:
                        row_text = ' | '.join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text:
                            paragraphs.append(row_text)
                    logger.info(f"   ✅ Table {table_num}: Extracted {len(table.rows)} rows")
            
            full_text = '\n\n'.join(paragraphs)
            logger.info("="*60)
            logger.info("✅ DOCX TEXT EXTRACTION COMPLETED")
            logger.info(f"   Total paragraphs: {len(paragraphs)}")
            logger.info(f"   Total characters: {len(full_text):,}")
            logger.info(f"   Total words (approx): {len(full_text.split()):,}")
            logger.info("="*60)
            
            return full_text
        
        except Exception as e:
            logger.error(f"❌ Failed to extract text from DOCX {file_name}: {e}", exc_info=True)
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            return None
