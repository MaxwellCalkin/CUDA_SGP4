"""Utilities for converting TLEs into GPU-friendly arrays."""

from cuda_sgp4.src.TLE import SafeTLEParser
import numpy as np
from datetime import datetime, timedelta
import warnings


# List of all attributes from ElsetRec class.  These are converted to
# columns for the GPU implementation.
ATTRIBUTES = [
    'whichconst', 'satnum', 'epochyr', 'epochtynumrev', 'error', 'operationmode',
    'init', 'method', 'a', 'altp', 'alta', 'epochdays', 'jdsatepoch', 'jdsatepochF',
    'nddot', 'ndot', 'bstar', 'rcse', 'inclo', 'nodeo', 'ecco', 'argpo', 'mo', 'no_kozai',
    'no_unkozai', 'classification', 'intldesg', 'ephtype', 'elnum', 'revnum',
    'gno_unkozai', 'am', 'em', 'im', 'Om', 'om', 'mm', 'nm', 't',
    'tumin', 'mu', 'radiusearthkm', 'xke', 'j2', 'j3', 'j4', 'j3oj2',
    'dia_mm', 'period_sec', 'active', 'not_orbital', 'rcs_m2',
    'ep', 'inclp', 'nodep', 'argpp', 'mp',
    'isimp', 'aycof', 'con41', 'cc1', 'cc4', 'cc5', 'd2', 'd3', 'd4', 'delmo', 'eta', 'argpdot',
    'omgcof', 'sinmao', 't2cof', 't3cof', 't4cof', 't5cof', 'x1mth2', 'x7thm1', 'mdot', 'nodedot',
    'xlcof', 'xmcof', 'nodecf',
    'irez', 'd2201', 'd2211', 'd3210', 'd3222', 'd4410', 'd4422', 'd5220', 'd5232',
    'd5421', 'd5433', 'dedt', 'del1', 'del2', 'del3', 'didt', 'dmdt', 'dnodt', 'domdt',
    'e3', 'ee2', 'peo', 'pgho', 'pho', 'pinco', 'plo', 'se2', 'se3', 'sgh2', 'sgh3',
    'sgh4', 'sh2', 'sh3', 'si2', 'si3', 'sl2', 'sl3', 'sl4', 'gsto', 'xfact', 'xgh2',
    'xgh3', 'xgh4', 'xh2', 'xh3', 'xi2', 'xi3', 'xl2', 'xl3', 'xl4', 'xlamo', 'zmol',
    'zmos', 'atime', 'xli', 'xni', 'snodm', 'cnodm', 'sinim', 'cosim', 'sinomm',
    'cosomm', 'day', 'emsq', 'gam', 'rtemsq', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
    'ss1', 'ss2', 'ss3', 'ss4', 'ss5', 'ss6', 'ss7', 'sz1', 'sz2', 'sz3', 'sz11',
    'sz12', 'sz13', 'sz21', 'sz22', 'sz23', 'sz31', 'sz32', 'sz33', 'z1', 'z2', 'z3',
    'z11', 'z12', 'z13', 'z21', 'z22', 'z23', 'z31', 'z32', 'z33', 'argpm', 'inclm',
    'nodem', 'dndt', 'eccsq',
    'ainv', 'ao', 'con42', 'cosio', 'cosio2', 'omeosq', 'posq', 'rp', 'rteosq', 'sinio',
]

WHICHCONST_IDX = ATTRIBUTES.index("whichconst")
T_IDX = ATTRIBUTES.index("t")


def _build_tle_array(tles, current_time):
    """Return a ``numpy`` array representation of ``tles``."""
    tle_arrays = np.zeros((len(tles), len(ATTRIBUTES)), dtype=np.float64)

    for i, tle in enumerate(tles):
        for j, attr in enumerate(ATTRIBUTES):
            value = getattr(tle.rec, attr, 0.0)
            if isinstance(value, str):
                tle_arrays[i, j] = ord(value) if len(value) == 1 else 0.0
            else:
                try:
                    tle_arrays[i, j] = float(value)
                except (TypeError, ValueError):
                    tle_arrays[i, j] = 0.0

        tle_arrays[i, WHICHCONST_IDX] = 2.0

        epoch_year = int(tle.rec.epochyr)
        epoch_day = tle.rec.epochdays
        epoch_year += 2000 if epoch_year < 57 else 1900
        epoch = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
        time_diff = (current_time - epoch).total_seconds() / 60
        tle_arrays[i, T_IDX] = time_diff

    return tle_arrays, tles


def initialize_tle_arrays_from_lines(tle_lines, current_time):
    """Create TLE objects from ``tle_lines`` and build the array.
    
    Uses SafeTLEParser for robust handling of real-world TLE format variations.
    """
    tles = []
    failed_indices = []
    
    for i, (l1, l2) in enumerate(tle_lines):
        try:
            tle = SafeTLEParser(l1, l2, strict=False)
            if tle.has_errors():
                # Log warnings but continue with parsed values
                warnings.warn(
                    f"TLE at index {i} had parsing errors but was recovered: "
                    f"{'; '.join(tle.get_errors())}"
                )
            tles.append(tle)
        except Exception as e:
            # Critical error - TLE could not be parsed at all
            warnings.warn(f"Failed to parse TLE at index {i}: {str(e)}")
            failed_indices.append(i)
    
    if failed_indices:
        warnings.warn(
            f"Failed to parse {len(failed_indices)} TLEs out of {len(tle_lines)}. "
            f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
        )
    
    if not tles:
        raise ValueError("No TLEs could be successfully parsed")
    
    return _build_tle_array(tles, current_time)


def initialize_tle_arrays(tle_lines, current_time):
    """Backward compatible wrapper for ``initialize_tle_arrays_from_lines``."""
    return initialize_tle_arrays_from_lines(tle_lines, current_time)
