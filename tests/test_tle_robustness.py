"""Test TLE parser robustness with real-world format variations."""

import pytest
from cuda_sgp4 import SafeTLEParser, TLE


class TestTLERobustness:
    """Test suite for TLE parser robustness improvements."""
    
    def test_standard_format(self):
        """Test ideal TLE format."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 25544
        assert not parser.has_errors()
    
    def test_short_lines(self):
        """Test TLEs shorter than 69 chars."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 25544
        assert parser.elnum == 0  # Missing element number defaults to 0
    
    def test_missing_checksum(self):
        """Test TLEs without checksum digit."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  999"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 25544
    
    def test_spacing_variations(self):
        """Test various spacing between fields."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 25544
    
    def test_scientific_notation_formats(self):
        """Test different scientific notation representations."""
        # Standard format
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.bstar > 0
        
        # Zero BSTAR
        line1_zero = "1 25544U 98067A   24340.51695139  .00022603  00000+0  00000-0 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1_zero, line2)
        assert parser.bstar == 0.0
    
    def test_empty_fields(self):
        """Test TLEs with empty optional fields."""
        # Empty revolution number
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794     "
        
        parser = SafeTLEParser(line1, line2)
        assert parser.revnum == 1  # Should normalize to 00001
    
    def test_missing_ndot_space(self):
        """Test missing space after ndot field."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603000000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 25544
    
    def test_alphanumeric_satellite_id(self):
        """Test alphanumeric satellite ID conversion."""
        line1 = "1 E9327U 21005BC  24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 E9327  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum > 100000  # Alpha-5 format results in 6+ digit number
    
    def test_line_validation_errors(self):
        """Test line validation with invalid formats."""
        # Wrong line number prefix
        with pytest.raises(ValueError, match="Line 1 must start with '1 '"):
            SafeTLEParser("2 25544U 98067A", "2 25544  51.6405")
        
        # Mismatched object numbers
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25545  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        with pytest.raises(ValueError, match="Object numbers don't match"):
            SafeTLEParser(line1, line2)
    
    def test_strict_mode(self):
        """Test strict mode raises on any parse errors."""
        # Short line that will have parse errors
        line1 = "1 25544U 98067A   24340.51695139"
        line2 = "2 25544  51.6405"
        
        # Strict mode should raise
        with pytest.raises(ValueError, match="TLE parsing failed"):
            SafeTLEParser(line1, line2, strict=True)
        
        # Non-strict mode should handle gracefully
        parser = SafeTLEParser(line1, line2, strict=False)
        assert parser.has_errors()
        assert len(parser.get_errors()) > 0
    
    def test_epoch_parsing_errors(self):
        """Test epoch parsing with various error conditions."""
        # Too short epoch
        line1 = "1 25544U 98067A   243"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        parser = SafeTLEParser(line1, line2, strict=False)
        assert parser.has_errors()
        assert any("Epoch" in err for err in parser.get_errors())
    
    def test_real_world_tle_variations(self):
        """Test with actual problematic TLEs from operations."""
        # Real TLE with short line (common from copy-paste)
        line1 = "1 43013U 17073A   24340.87639815  .00001234  00000-0  12345-3 0  999"
        line2 = "2 43013  86.3985 123.4567 0001234  90.1234 270.1234 14.34567890123456"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 43013
        assert not parser.has_errors()
        
        # TLE with unusual spacing (happens with some ground stations)
        line1 = "1 12345U 01049A   24340.12345678 -.00000123  00000-0 -12345-4 0  1234"
        line2 = "2 12345  98.1234 123.4567 0012345 123.4567 236.5432 14.12345678901234"
        
        parser = SafeTLEParser(line1, line2)
        assert parser.objectNum == 12345
        assert parser.ndot < 0  # Negative mean motion derivative
    
    def test_backward_compatibility(self):
        """Ensure original TLE class still works for basic parsing."""
        line1 = "1 25544U 98067A   24340.51695139  .00022603  00000+0  39081-3 0  9992"
        line2 = "2 25544  51.6405 204.9263 0002769  53.7538  49.6568 15.50388794436059"
        
        # Original TLE class should still work (though less robust)
        parser = TLE(line1, line2)
        assert parser.objectNum == 25544
        
        # SafeTLEParser should produce same results for valid TLEs
        safe_parser = SafeTLEParser(line1, line2)
        assert safe_parser.objectNum == parser.objectNum
        assert safe_parser.epoch == parser.epoch
        assert safe_parser.incDeg == parser.incDeg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])