"""Demo of the improved TLE parser handling real-world format variations."""

from datetime import datetime
import warnings
from cuda_sgp4 import SafeTLEParser, cuda_sgp4


def demo_safetleparser():
    """Demonstrate SafeTLEParser handling various TLE format issues."""
    
    print("=== SafeTLEParser Demo ===\n")
    
    # Example 1: Short TLE lines (common from copy-paste)
    print("1. Handling short TLE lines:")
    line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0"
    line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794"
    
    parser = SafeTLEParser(line1, line2)
    print(f"   Object number: {parser.objectNum}")
    print(f"   Has errors: {parser.has_errors()}")
    if parser.has_errors():
        print(f"   Errors: {parser.get_errors()}")
    print()
    
    # Example 2: Missing space after ndot field
    print("2. Handling missing space after ndot:")
    line1 = "1 25544U 98067A   24340.51695139  .00022603000000+0  39081-3 0  9992"
    line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
    
    parser = SafeTLEParser(line1, line2)
    print(f"   Object number: {parser.objectNum}")
    print(f"   Mean motion derivative: {parser.ndot}")
    print()
    
    # Example 3: Empty revolution number
    print("3. Handling empty revolution number:")
    line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
    line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794     "
    
    parser = SafeTLEParser(line1, line2)
    print(f"   Revolution number: {parser.revnum}")
    print()


def demo_cuda_sgp4_integration():
    """Demonstrate cuda_sgp4 with problematic TLEs."""
    
    print("=== cuda_sgp4 Integration Demo ===\n")
    
    try:
        from numba import cuda
        if not cuda.is_available():
            print("CUDA is not available. Skipping GPU demo.")
            return
    except Exception as e:
        print(f"CUDA check failed: {e}. Skipping GPU demo.")
        return
    
    # Mix of good and problematic TLEs
    tle_lines = [
        # Good TLE
        ("1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992",
         "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"),
        
        # Short lines
        ("1 43013U 17073A   24340.87639815  .00001234  00000-0  12345-3 0",
         "2 43013  86.3985 123.4567 0001234  90.1234 270.1234 14.34567890"),
    ]
    
    start_time = datetime(2024, 12, 1)
    
    print("Processing TLEs with cuda_sgp4...")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            positions, velocities = cuda_sgp4(
                tle_lines,
                timestep_length_in_seconds=60,
                total_sim_seconds=180,  # 3 minutes
                start_time=start_time,
                return_device_arrays=False
            )
            
            print(f"Successfully processed {len(tle_lines)} TLEs")
            print(f"Output shape: {positions.shape}")
            
            if w:
                print(f"\nWarnings generated ({len(w)} total):")
                for warning in w[:3]:  # Show first 3 warnings
                    print(f"  - {warning.message}")
    except Exception as e:
        print(f"CUDA execution failed: {type(e).__name__}: {str(e)[:100]}...")
        print("This is likely due to CUDA environment issues, not the TLE parser.")


def demo_strict_mode():
    """Demonstrate strict mode for validation."""
    
    print("\n=== Strict Mode Demo ===\n")
    
    # Problematic TLE
    line1 = "1 25544U 98067A   24340.51695139"  # Too short
    line2 = "2 25544  51.6405"  # Too short
    
    print("Attempting to parse with strict=True:")
    try:
        parser = SafeTLEParser(line1, line2, strict=True)
    except ValueError as e:
        print(f"  Caught error: {e}")
    
    print("\nParsing with strict=False (tolerant mode):")
    parser = SafeTLEParser(line1, line2, strict=False)
    print(f"  Object number: {parser.objectNum}")
    print(f"  Errors encountered: {len(parser.get_errors())}")


def main():
    """Run all demos."""
    demo_safetleparser()
    demo_cuda_sgp4_integration()
    demo_strict_mode()
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()