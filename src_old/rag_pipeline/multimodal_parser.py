import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import logging


class CybersecurityDocumentParser:
    """Extract multimodal content from cybersecurity PDFs"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: str) -> Dict:
        """Extract all content types from PDF"""
        doc = fitz.open(pdf_path)

        result = {
            'text_content': [],
            'images': [],
            'tables': [],
            'metadata': self._extract_metadata(doc),
            'structured_data': {
                'iocs': [],
                'cves': [],
                'yara_rules': [],
                'network_diagrams': []
            }
        }

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            text = page.get_text()
            result['text_content'].append({
                'page': page_num,
                'text': text,
                'bbox': page.rect
            })

            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                image_data = self._extract_image(doc, img, page_num, img_index)
                if image_data:
                    result['images'].append(image_data)

            # Extract tables
            tables = self._extract_tables_from_page(page)
            result['tables'].extend(tables)

            # Extract structured cybersecurity data
            structured = self._extract_structured_data(text)
            for key, values in structured.items():
                result['structured_data'][key].extend(values)

        doc.close()
        return result

    def _extract_image(self, doc, img, page_num: int, img_index: int) -> Dict:
        """Extract and process individual image"""
        try:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")

                # Convert to PIL Image for processing
                pil_image = Image.open(io.BytesIO(img_data))

                # OCR on image
                ocr_text = pytesseract.image_to_string(pil_image)

                # Detect if it's a network diagram or security-related image
                image_type = self._classify_image_type(pil_image, ocr_text)

                result = {
                    'page': page_num,
                    'index': img_index,
                    'image_data': img_data,
                    'ocr_text': ocr_text,
                    'type': image_type,
                    'size': pil_image.size
                }

                pix = None
                return result

        except Exception as e:
            self.logger.error(f"Error extracting image: {e}")

        return None

    def _extract_tables_from_page(self, page) -> List[Dict]:
        """Extract tables using table detection"""
        tables = []

        # Get table-like structures
        tabs = page.find_tables()

        for tab_index, tab in enumerate(tabs):
            try:
                df = tab.to_pandas()
                table_data = {
                    'page': page.number,
                    'index': tab_index,
                    'data': df.to_dict('records'),
                    'bbox': tab.bbox,
                    'columns': df.columns.tolist()
                }
                tables.append(table_data)
            except Exception as e:
                self.logger.error(f"Error extracting table: {e}")

        return tables

    def _extract_structured_data(self, text: str) -> Dict[str, List]:
        """Extract IoCs, CVEs, YARA rules from text"""
        import re

        structured = {
            'iocs': [],
            'cves': [],
            'yara_rules': [],
            'mitre_techniques': []
        }

        # CVE pattern
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        structured['cves'] = re.findall(cve_pattern, text)

        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)

        # Domain names
        domain_pattern = r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, text)

        # Hash values (MD5, SHA1, SHA256)
        hash_patterns = {
            'md5': r'\b[a-fA-F0-9]{32}\b',
            'sha1': r'\b[a-fA-F0-9]{40}\b',
            'sha256': r'\b[a-fA-F0-9]{64}\b'
        }

        hashes = []
        for hash_type, pattern in hash_patterns.items():
            found_hashes = re.findall(pattern, text)
            hashes.extend([{'type': hash_type, 'value': h}
                          for h in found_hashes])

        # MITRE ATT&CK techniques
        mitre_pattern = r'T\d{4}(?:\.\d{3})?'
        structured['mitre_techniques'] = re.findall(mitre_pattern, text)

        # Combine IoCs
        structured['iocs'] = {
            'ips': ips,
            'domains': domains,
            'hashes': hashes
        }

        return structured

    def _classify_image_type(self, image: Image, ocr_text: str) -> str:
        """Classify image type (network diagram, architecture, etc.)"""
        # Simple classification based on OCR text and image properties
        text_lower = ocr_text.lower()

        network_keywords = ['firewall', 'router',
                            'switch', 'network', 'topology', 'lan', 'wan']
        architecture_keywords = ['architecture', 'diagram', 'flow', 'process']

        if any(keyword in text_lower for keyword in network_keywords):
            return 'network_diagram'
        elif any(keyword in text_lower for keyword in architecture_keywords):
            return 'architecture_diagram'
        elif image.size[0] > image.size[1] and 'table' in text_lower:
            return 'table_image'
        else:
            return 'general'
