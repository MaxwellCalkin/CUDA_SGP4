from cuda_sgp4.src.TLE import TLE
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def initialize_tle_arrays(path, current_time):
    df = pd.read_csv(path)

    # Convert 'epoch' column to datetime and filter out TLEs older than 2 months
    df['epoch'] = pd.to_datetime(df['epoch'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # two_months_ago = current_time - timedelta(days=60)
    # df = df[df['epoch'] > two_months_ago]

    # Create TLE objects for each row in the filtered DataFrame
    tles = [TLE(row['line1'], row['line2']) for index, row in df.iterrows()]
    print(tles[0].rec)

    # List of all attributes from ElsetRec class
    attributes = [
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

    tle_arrays = np.zeros((len(tles), len(attributes)), dtype=np.float64)

    for i, tle in enumerate(tles):
        for j, attr in enumerate(attributes):
            value = getattr(tle.rec, attr, None)
            if value is None:
                tle_arrays[i, j] = 0.0
            elif isinstance(value, str):
                # Handle string attributes
                if len(value) > 1:
                    tle_arrays[i, j] = 0.0
                elif len(value) < 1:
                    tle_arrays[i, j] = 0.0
                else:
                    tle_arrays[i, j] = ord(value)
            else:
                try:
                    tle_arrays[i, j] = float(value)
                except (TypeError, ValueError):
                    tle_arrays[i, j] = 0.0  # Default to 0.0 if conversion fails

        # Set 'whichconst' to 2 (SGP4.wgs72)
        tle_arrays[i, attributes.index('whichconst')] = 2.0

        # Compute 't', time since epoch in minutes
        epoch_year = int(tle.rec.epochyr)
        epoch_day = tle.rec.epochdays

        if epoch_year < 57:
            epoch_year += 2000
        else:
            epoch_year += 1900

        epoch = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
        time_diff = (current_time - epoch).total_seconds() / 60  # Convert time_diff to minutes
        tle_arrays[i, attributes.index('t')] = time_diff

    print(f"Number of TLEs after filtering: {len(tles)}")
    return tle_arrays, tles
