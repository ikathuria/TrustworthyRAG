import time
import shutil
from pathlib import Path
from src.preprocessing.document_parser import DocumentParser
import src.utils.constants as C

def benchmark():
    # Setup
    test_pdf = C.TEST_PDF
    doc_name = Path(test_pdf).stem
    cached_path = Path(C.PROCESSED_DATA_DIR) / f"{doc_name}_parsed.pkl"
    
    # Ensure clean state
    if cached_path.exists():
        cached_path.unlink()
    
    print(f"Benchmarking parsing for: {test_pdf}")
    
    # Run 1: Cold Start (Parsing)
    print("\n--- Run 1: Cold Start (Parsing) ---")
    start_time = time.time()
    parser = DocumentParser(config={"dtype": "auto", "device_map": "auto"})
    results1 = parser.parse_batch([test_pdf])
    end_time = time.time()
    cold_time = end_time - start_time
    print(f"Cold start time: {cold_time:.2f} seconds")
    
    # Verify Run 1
    if not results1:
        print("Error: No results from Run 1")
        return
    
    # Run 2: Warm Start (Caching)
    print("\n--- Run 2: Warm Start (Caching) ---")
    start_time = time.time()
    # Re-initialize parser to simulate a fresh run
    parser = DocumentParser(config={"dtype": "auto", "device_map": "auto"})
    results2 = parser.parse_batch([test_pdf])
    end_time = time.time()
    warm_time = end_time - start_time
    print(f"Warm start time: {warm_time:.2f} seconds")
    
    # Verify Run 2
    if not results2:
        print("Error: No results from Run 2")
        return
        
    # Assertions
    print("\n--- Results ---")
    print(f"Speedup: {cold_time / warm_time:.2f}x")
    
    if warm_time < cold_time:
        print("SUCCESS: Warm start is faster.")
    else:
        print("FAILURE: Warm start is not faster.")
        
    # Verify content match (basic check)
    if len(results1) == len(results2):
        print("SUCCESS: Result count matches.")
    else:
        print("FAILURE: Result count mismatch.")

if __name__ == "__main__":
    benchmark()
