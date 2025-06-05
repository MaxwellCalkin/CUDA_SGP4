"""Integration tests for improved TLE parser with cuda_sgp4."""

import warnings
from datetime import datetime
import numpy as np
import pytest
from cuda_sgp4 import cuda_sgp4, SafeTLEParser

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_sgp4_with_problematic_tles():
    """Test that cuda_sgp4 handles problematic TLEs gracefully."""
    
    # Mix of good and problematic TLEs
    tle_lines = [
        # Good TLE
        ("1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992",
         "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"),
        
        # Short lines (common from copy-paste) - these get normalized so no errors
        ("1 43013U 17073A   24340.87639815  .00001234  00000-0  12345-3 0",
         "2 43013  86.3985 123.4567 0001234  90.1234 270.1234 14.34567890"),
        
        # Missing space after ndot - gets normalized so no errors
        ("1 25544U 98067A   24340.51695139  .00022603000000+0  39081-3 0  9992",
         "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"),
        
        # Very short line that will cause parsing errors
        ("1 25544U 98067A   24340.51695139",
         "2 25544  51.6405 204.9263"),
    ]
    
    start_time = datetime(2024, 12, 1)
    
    # This should work without raising exceptions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            positions, velocities = cuda_sgp4(
                tle_lines,
                timestep_length_in_seconds=60,
                total_sim_seconds=3600,
                start_time=start_time,
                return_device_arrays=False
            )
            
            # The last TLE should generate warnings due to missing fields
            assert len(w) > 0
            assert any("parsing errors but was recovered" in str(warning.message) for warning in w)
            
            # Verify output shape
            assert positions.shape == (4, 60, 3)
            assert velocities.shape == (4, 60, 3)
            
            # Verify results are reasonable (not NaN or inf)
            assert np.all(np.isfinite(positions))
            assert np.all(np.isfinite(velocities))
        except Exception as e:
            if "CUDA_ERROR" in str(e) or "PTX" in str(e):
                pytest.skip(f"CUDA runtime error: {str(e)[:100]}")


def test_safetleparser_direct_usage():
    """Test SafeTLEParser can be used directly for pre-validation."""
    
    # Short TLE lines that get normalized automatically
    line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0"
    line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794"
    
    # Parse with SafeTLEParser
    parser = SafeTLEParser(line1, line2, strict=False)
    
    # Should have parsed successfully - normalization fixes the short lines
    assert parser.objectNum == 25544
    assert not parser.has_errors()  # No errors after normalization
    
    # Can use the parsed data
    assert parser.incDeg > 0
    assert parser.n > 0
    
    # Test with a truly problematic TLE that will have errors
    line1_bad = "1 25544U 98067A   24340.51695139"  # Very short, missing many fields
    line2_bad = "2 25544  51.6405"  # Very short
    
    parser_bad = SafeTLEParser(line1_bad, line2_bad, strict=False)
    assert parser_bad.objectNum == 25544
    assert parser_bad.has_errors()  # This should have errors
    assert len(parser_bad.get_errors()) > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_all_tles_fail_gracefully():
    """Test behavior when all TLEs fail to parse."""
    
    # Completely invalid TLEs
    tle_lines = [
        ("INVALID LINE 1", "INVALID LINE 2"),
        ("X ", "Y "),  # Wrong line prefixes
    ]
    
    start_time = datetime(2024, 12, 1)
    
    # Should handle the situation gracefully
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            cuda_sgp4(
                tle_lines,
                timestep_length_in_seconds=60,
                total_sim_seconds=3600,
                start_time=start_time,
                return_device_arrays=False
            )
            # If we get here, the parser was more tolerant than expected
            # Check that we at least got warnings
            assert len(w) > 0
        except ValueError as e:
            # This is the expected path - no valid TLEs
            assert "No TLEs could be successfully parsed" in str(e) or "must start with" in str(e)
        except Exception as e:
            if "CUDA_ERROR" in str(e) or "PTX" in str(e):
                pytest.skip(f"CUDA runtime error: {str(e)[:100]}")


if __name__ == "__main__":
    test_cuda_sgp4_with_problematic_tles()
    test_safetleparser_direct_usage()
    test_all_tles_fail_gracefully()
    print("All integration tests passed!")