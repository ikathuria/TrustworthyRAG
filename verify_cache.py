import time
from pathlib import Path
from src.preprocessing.document_parser import DocumentParser
import src.utils.constants as C

def verify_cache():
    test_pdf = C.TEST_PDF
    doc_name = Path(test_pdf).stem
    cached_path = Path(C.PROCESSED_DATA_DIR) / f"{doc_name}_parsed.pkl"
    
    if not cached_path.exists():
        print(f"Error: Cache file not found at {cached_path}")
        return

    print(f"Verifying cache for: {test_pdf}")
    print(f"Cache file size: {cached_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Run: Warm Start (Caching)
    print("\n--- Warm Start (Caching) ---")
    start_time = time.time()
    parser = DocumentParser(config={"dtype": "auto", "device_map": "auto"})
    results = parser.parse_batch([test_pdf])
    end_time = time.time()
    warm_time = end_time - start_time
    
    print(f"Warm start time: {warm_time:.4f} seconds")
    
    if results:
        print(f"Successfully loaded {len(results)} document(s).")
        print(f"Source file: {results[0].source_file}")
        print(f"Text length: {len(results[0].text.get('content', ''))}")
    else:
        print("Error: No results loaded.")

if __name__ == "__main__":
    verify_cache()
