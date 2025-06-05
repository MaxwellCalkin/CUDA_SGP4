import math
from .ElsetRec import ElsetRec
import cuda_sgp4.src.SGP4 as SGP4
from typing import Optional, Tuple
class TLENormalizer:
    """Normalize TLE lines to standard format before parsing."""
    
    @staticmethod
    def normalize_line(line: str, line_num: int) -> str:
        """Normalize a TLE line to standard 69-character format."""
        line = line.rstrip()
        
        if len(line) < 69:
            line = line.ljust(69)
        elif len(line) > 69:
            line = line[:69]
            
        return line
    
    @staticmethod
    def normalize_line1(line: str) -> str:
        """Ensure Line 1 meets parser expectations."""
        line = TLENormalizer.normalize_line(line, 1)
        
        if len(line) >= 44:
            if line[43] != ' ' and line[43] in '0123456789+-':
                line = line[:44] + ' ' + line[44:]
                
        return line
    
    @staticmethod
    def normalize_line2(line: str) -> str:
        """Ensure Line 2 meets parser expectations."""
        line = TLENormalizer.normalize_line(line, 2)
        
        if len(line) >= 68:
            rev_num = line[63:68].strip()
            if not rev_num:
                line = line[:63] + '00001' + (line[68:] if len(line) > 68 else '')
                
        return line
class TLE:
    def __init__(self, line1="", line2=""):
        self.rec = None
        
        self.line1 = None
        self.line2 = None
        
        self.intlid = None
        self.objectNum = 0
        self.epoch = None
        self.ndot = 0
        self.nddot = 0
        self.bstar = 0
        self.elnum = 0
        self.incDeg = 0
        self.raanDeg = 0
        self.ecc = 0
        self.argpDeg = 0
        self.maDeg = 0
        self.n = 0
        self.revnum = 0
        
        self.parseErrors = None
        
        self.sgp4Error = 0
        
        self.parseLines(line1, line2)


    def getRVForDate(self, t):
        t -= self.epoch
        t /= 60000
        
        return self.getRV(t)

    def getRV(self, minutesAfterEpoch):
        r = [0, 0, 0]
        v = [0, 0, 0]
        
        self.rec.error = 0
        SGP4.sgp4(self.rec, minutesAfterEpoch, r, v)
        self.sgp4Error = self.rec.error
        return [r, v]

    
    def parseLines(self, line1, line2):
        """Parse TLE lines with error handling and validation."""
        self.parseErrors = []
        self.rec = ElsetRec()

        self.line1 = line1
        self.line2 = line2
        
        try:
            self.objectNum = int(gd(line1, 2, 7))
        except Exception as e:
            self.parseErrors.append(f"Object number (cols 3-7): {str(e)}")
            self.objectNum = 0
        
        if len(line1) > 7:
            self.rec.classification = line1[7]
        else:
            self.rec.classification = 'U'
            self.parseErrors.append("Classification (col 8): Line too short")

        try:
            self.intlid = safe_substring(line1, 9, 17).strip()
        except Exception as e:
            self.parseErrors.append(f"International designator (cols 10-17): {str(e)}")
            self.intlid = ""
            
        try:
            self.epoch = self.parseEpoch(safe_substring(line1, 18, 32).strip())
        except Exception as e:
            self.parseErrors.append(f"Epoch (cols 19-32): {str(e)}")
            self.epoch = 0

        try:
            self.ndot = gdi(safe_char(line1, 33, ' '), line1, 35, 44)
        except Exception as e:
            self.parseErrors.append(f"Mean motion derivative (cols 34-43): {str(e)}")
            self.ndot = 0

        try:
            self.nddot = gdi(safe_char(line1, 44, ' '), line1, 45, 50)
            exp_str = safe_substring(line1, 50, 52)
            if exp_str.strip():
                exp = float(exp_str)
                self.nddot *= math.pow(10.0, exp)
        except Exception as e:
            self.parseErrors.append(f"Mean motion 2nd derivative (cols 45-52): {str(e)}")
            self.nddot = 0

        try:
            self.bstar = gdi(safe_char(line1, 53, ' '), line1, 54, 59)
            exp_str = safe_substring(line1, 59, 61)
            if exp_str.strip():
                exp = float(exp_str)
                self.bstar *= math.pow(10.0, exp)
        except Exception as e:
            self.parseErrors.append(f"BSTAR (cols 54-61): {str(e)}")
            self.bstar = 0

        try:
            self.elnum = int(gd(line1, 64, 68))
        except Exception as e:
            self.parseErrors.append(f"Element number (cols 65-68): {str(e)}")
            self.elnum = 0
        
        try:
            self.incDeg = gd(line2, 8, 16)
        except Exception as e:
            self.parseErrors.append(f"Inclination (cols 9-16): {str(e)}")
            self.incDeg = 0
            
        try:
            self.raanDeg = gd(line2, 17, 25)
        except Exception as e:
            self.parseErrors.append(f"RAAN (cols 18-25): {str(e)}")
            self.raanDeg = 0
            
        try:
            self.ecc = gdi('+', line2, 26, 33)
        except Exception as e:
            self.parseErrors.append(f"Eccentricity (cols 27-33): {str(e)}")
            self.ecc = 0
            
        try:
            self.argpDeg = gd(line2, 34, 42)
        except Exception as e:
            self.parseErrors.append(f"Argument of perigee (cols 35-42): {str(e)}")
            self.argpDeg = 0
            
        try:
            self.maDeg = gd(line2, 43, 51)
        except Exception as e:
            self.parseErrors.append(f"Mean anomaly (cols 44-51): {str(e)}")
            self.maDeg = 0
        
        try:
            self.n = gd(line2, 52, 63)
        except Exception as e:
            self.parseErrors.append(f"Mean motion (cols 53-63): {str(e)}")
            self.n = 0
        
        try:
            rev_str = safe_substring(line2, 63, 68).strip()
            if rev_str:
                self.revnum = int(float(rev_str))
            else:
                self.revnum = 0
        except Exception as e:
            self.parseErrors.append(f"Revolution number (cols 64-68): {str(e)}")
            self.revnum = 0
        
        self.setValsToRec()

    def setValsToRec(self):
        deg2rad = math.pi / 180.0         # //   0.0174532925199433
        xpdotp = 1440.0 / (2.0 * math.pi) # // 229.1831180523293

        self.rec.elnum = self.elnum
        self.rec.revnum = self.revnum
        self.rec.satnum = self.objectNum
        self.rec.bstar = self.bstar
        self.rec.inclo = self.incDeg*deg2rad
        self.rec.nodeo = self.raanDeg*deg2rad
        self.rec.argpo = self.argpDeg*deg2rad
        self.rec.mo = self.maDeg*deg2rad
        self.rec.ecco = self.ecc
        self.rec.no_kozai = self.n/xpdotp
        self.rec.ndot = self.ndot / (xpdotp*1440.0)
        self.rec.nddot = self.nddot / (xpdotp*1440.0*1440.0)
        
        SGP4.sgp4init(ord('a'), self.rec)


    
    def parseEpoch(self, str):
        """Parse the TLE epoch format to a date with error handling."""
        try:
            if len(str) < 14:
                raise ValueError(f"Epoch string too short: {len(str)} chars")
                
            year = int(str[0:2].strip())
            self.rec.epochyr = year
            if year > 56:
                year += 1900
            else:
                year += 2000

            doy = int(str[2:5].strip())
            dfrac_str = str[5:].strip()
            if dfrac_str:
                dfrac = float("0" + dfrac_str)
            else:
                dfrac = 0.0
                
            self.rec.epochdays = doy
            self.rec.epochdays += dfrac
            dfrac *= 24.0
            hr = int(dfrac)
            dfrac = 60.0 * (dfrac - hr)
            mn = int(dfrac)
            dfrac = 60.0 * (dfrac - mn)
            sec = dfrac
            sc = int(sec)
            micro = 1000000 * (dfrac - sc)

            mon = 0
            dys = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if isLeap(year):
                dys[1] = 29

            while mon < 12 and dys[mon] < doy:
                doy -= dys[mon]
                mon += 1

            mon += 1
            day = doy

            jd = SGP4.jday(year, mon, day, hr, mn, sec)
            self.rec.jdsatepoch = jd[0]
            self.rec.jdsatepochF = jd[1]
            ep = jd[0] - 2440587.5  # 1970
            ep = int(ep)
            ep = ep + jd[1]
            ep = ep * 86400.0
            ep = ep * 1000

            return ep
        except Exception as e:
            raise ValueError(f"Failed to parse epoch '{str}': {str(e)}")
 
    #/**
    # * parse a double from the substring.
    # * 
    # * @param str
    # * @param start
    # * @param end
    # * @return
    # */
def safe_substring(s: str, start: int, end: int) -> str:
    """Safely extract substring with bounds checking."""
    if len(s) < end:
        if len(s) <= start:
            return ""
        end = len(s)
    return s[start:end]

def safe_char(s: str, pos: int, default: str = ' ') -> str:
    """Safely extract a character with bounds checking."""
    if len(s) > pos:
        return s[pos]
    return default

def gd(str, start, end):
    return gd2(str, start, end, 0)
   

 
    #/**
    # * parse a double from the substring.
    # * 
    # * @param str
    # * @param start
    # * @param end
    # * @param defVal
    # * @return
    # */


def gd2(str, start, end, defVal):
    """Parse a double from substring with bounds checking and error handling."""
    if len(str) < end:
        if len(str) <= start:
            return defVal
        end = len(str)
    
    substr = str[start:end].strip()
    
    if not substr:
        return defVal
    
    if substr[0].isalpha():
        alpha = substr[0].upper()
        numeric = substr[1:]
        if alpha in 'IO':
            return defVal
        elif alpha <= 'H':
            base = 10
        elif alpha <= 'N':
            base = 9
        elif alpha <= 'Z':
            base = 8
        else:
            return defVal

        try:
            alpha_value = ord(alpha) - ord('A') + base
            num = alpha_value * 100000 + float(numeric)
        except (ValueError, IndexError):
            return defVal
    else:
        try:
            num = float(substr)
        except ValueError:
            return defVal
    return num

def gdi(sign, str, start,  end):
      return gdi2(sign,str,start,end,0)
    
   # /**
   #  * parse a double from the substring after adding an implied decimal point.
   #  * 
   #  * @param str
   #  * @param start
   #  * @param end
   #  * @param defVal
   #  * @return
   #  */
def gdi2(sign, str, start, end, defVal):
    """Parse a decimal with implied decimal point and bounds checking."""
    if len(str) < end:
        if len(str) <= start:
            return defVal
        end = len(str)
    
    try:
        substr = str[start:end].strip()
        if not substr or substr == '00000-0':
            return 0.0
        num = float("0." + substr)
        if sign == '-':
            num *= -1.0
        return num
    except (ValueError, IndexError):
        return defVal


def isLeap(year):
    if year % 4 > 0:
        return False

    if year % 100 > 0:
        return True

    if year % 400 > 0:
        return False

    return True


class SafeTLEParser(TLE):
    """Backward-compatible TLE parser with safety features."""
    
    def __init__(self, line1: str, line2: str, strict: bool = False):
        """Initialize SafeTLEParser.
        
        Args:
            line1: First line of TLE
            line2: Second line of TLE
            strict: If True, raise exception on any parse errors. If False, use defaults.
        """
        self.strict = strict
        
        line1 = TLENormalizer.normalize_line1(line1)
        line2 = TLENormalizer.normalize_line2(line2)
        
        self._validate_lines(line1, line2)
        
        # Initialize parent but skip SGP4 init if critical values are missing
        self._safe_init(line1, line2)
        
        if self.strict and self.parseErrors:
            raise ValueError(f"TLE parsing failed: {'; '.join(self.parseErrors)}")
    
    def _safe_init(self, line1: str, line2: str):
        """Initialize TLE safely, avoiding SGP4 init if critical values are missing."""
        # Initialize all attributes first
        self.rec = None
        self.line1 = None
        self.line2 = None
        self.intlid = None
        self.objectNum = 0
        self.epoch = None
        self.ndot = 0
        self.nddot = 0
        self.bstar = 0
        self.elnum = 0
        self.incDeg = 0
        self.raanDeg = 0
        self.ecc = 0
        self.argpDeg = 0
        self.maDeg = 0
        self.n = 0
        self.revnum = 0
        self.parseErrors = None
        self.sgp4Error = 0
        
        # Call parent's parseLines
        super().parseLines(line1, line2)
        
    def setValsToRec(self):
        """Override to safely handle SGP4 initialization."""
        # Check if we have critical values needed for SGP4 initialization
        if self.n <= 0:
            self.parseErrors.append("Mean motion is zero or invalid - SGP4 initialization skipped")
            return
            
        # Call parent's setValsToRec only if safe
        super().setValsToRec()
    
    def _validate_lines(self, line1: str, line2: str):
        """Basic validation before parsing."""
        if not line1 or len(line1) < 2:
            raise ValueError("Line 1 is empty or too short")
            
        if not line1.startswith('1 '):
            raise ValueError(f"Line 1 must start with '1 ', got '{line1[:2]}'")
            
        if not line2 or len(line2) < 2:
            raise ValueError("Line 2 is empty or too short")
            
        if not line2.startswith('2 '):
            raise ValueError(f"Line 2 must start with '2 ', got '{line2[:2]}'")
        
        if len(line1) >= 7 and len(line2) >= 7:
            obj1 = safe_substring(line1, 2, 7).strip()
            obj2 = safe_substring(line2, 2, 7).strip()
            if obj1 and obj2 and obj1 != obj2:
                raise ValueError(f"Object numbers don't match: {obj1} != {obj2}")
    
    def has_errors(self) -> bool:
        """Check if any parsing errors occurred."""
        return bool(self.parseErrors)
    
    def get_errors(self) -> list:
        """Get list of parsing errors."""
        return self.parseErrors if self.parseErrors else []


