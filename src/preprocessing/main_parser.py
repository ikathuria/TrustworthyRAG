"""
Process all the raw documents in DATA_DIR using DocumentParser.
"""

from src.preprocessing.document_parser import DocumentParser


def main(file_paths):
    parser = DocumentParser()
    return parser.parse_batch(file_paths)


if __name__ == "__main__":
    file_paths = [
        "data/raw/WickedRose_andNCPH.pdf",
	]
    parsed_contents = main(file_paths)
