# Document Parsing with MinerU2.5

We utilized the MinerU2.5 tool for parsing documents in our preprocessing pipeline. MinerU2.5 is a powerful document analysis and extraction tool that helps in converting unstructured documents into structured data formats. It supports a wide range of document types, including PDFs, Word documents, and scanned images, making it suitable for diverse datasets.

## Features of MinerU2.5
- **Multi-format Support**: Capable of handling various document formats.
- **Advanced OCR**: Equipped with Optical Character Recognition (OCR) capabilities for extracting text from scanned documents.
- **Metadata Extraction**: Extracts metadata such as author, title, and creation date.
- **Customizable Parsing Rules**: Allows users to define custom rules for parsing specific document structures.

## Script Usage
To preprocess documents using MinerU2.5, we have created a class `DocumentParser` in our preprocessing script. Below is a brief overview of how to use this class:

```python
from document_parser import DocumentParser

# Initialize the DocumentParser
parser = DocumentParser()

# Parse a document
parsed_data = parser.parse("path/to/document.pdf")

# Access the extracted information
print(parsed_data["text"])
print(parsed_data["images"])
print(parsed_data["metadata"])
```
