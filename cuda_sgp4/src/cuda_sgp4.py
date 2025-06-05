# Imports
"""CUDA kernels and helpers for the SGP4 propagator."""

from numba import cuda, int32
import math

# Constants
pi = math.pi
twopi = 2.0 * pi
deg2rad = pi / 180.0

wgs72old = 1
wgs72 = 2
wgs84 = 3

def get_optimal_launch_config(num_satellites, min_blocks=8):
    """
    Calculate optimal CUDA launch configuration for good GPU utilization.
    
    Args:
        num_satellites: Number of satellites to process
        min_blocks: Minimum number of blocks to ensure good occupancy
        
    Returns:
        tuple: (blocks_per_grid, threads_per_block)
    """
    if not cuda.is_available():
        return 1, 1
    
    # Get GPU properties
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    multiprocessor_count = device.MULTIPROCESSOR_COUNT
    
    # Typical good choices for threads per block (powers of 2, multiples of warp size)
    preferred_block_sizes = [32, 64, 128, 256, 512, 1024]
    
    # Filter by what the device supports
    valid_block_sizes = [size for size in preferred_block_sizes if size <= max_threads_per_block]
    
    best_config = None
    best_utilization = 0
    
    for threads_per_block in valid_block_sizes:
        blocks_needed = (num_satellites + threads_per_block - 1) // threads_per_block
        
        # Ensure minimum number of blocks for good occupancy
        if blocks_needed < min_blocks:
            # Reduce threads per block to increase number of blocks
            threads_per_block = max(32, (num_satellites + min_blocks - 1) // min_blocks)
            # Round down to nearest multiple of 32 (warp size)
            threads_per_block = (threads_per_block // 32) * 32
            if threads_per_block == 0:
                threads_per_block = 32
            blocks_needed = (num_satellites + threads_per_block - 1) // threads_per_block
        
        # Calculate utilization metrics
        total_threads = blocks_needed * threads_per_block
        thread_utilization = num_satellites / total_threads if total_threads > 0 else 0
        
        # Prefer configurations that use more SMs
        sm_utilization = min(blocks_needed / multiprocessor_count, 1.0)
        
        # Combined score (weight thread efficiency more heavily)
        utilization_score = 0.7 * thread_utilization + 0.3 * sm_utilization
        
        if utilization_score > best_utilization:
            best_utilization = utilization_score
            best_config = (blocks_needed, threads_per_block)
    
    if best_config is None:
        # Fallback to simple calculation
        threads_per_block = min(256, max_threads_per_block)
        blocks_per_grid = max(min_blocks, (num_satellites + threads_per_block - 1) // threads_per_block)
        best_config = (blocks_per_grid, threads_per_block)
    
    return best_config

def suppress_cuda_warnings():
    """
    Suppress CUDA performance warnings for small test cases.
    
    Call this function before running CUDA kernels if you want to suppress
    NumbaPerformanceWarning messages for small workloads.
    """
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

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

# Create index mapping for attributes
index_mapping = {attr: idx for idx, attr in enumerate(attributes)}

# Export ``<attribute>Idx`` constants for compatibility
globals().update({f"{attr}Idx": idx for idx, attr in enumerate(attributes)})


# Device function dpper
@cuda.jit(device=True)
def dpper(tle_array, init, opsmode):
    """Apply deep space periodic perturbations."""
    # Constants
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490
    twopi = 2.0 * math.pi

    # Extract variables from tle_array using their indexes
    e3 = tle_array[e3Idx]
    ee2 = tle_array[ee2Idx]
    peo = tle_array[peoIdx]
    pgho = tle_array[pghoIdx]
    pho = tle_array[phoIdx]
    pinco = tle_array[pincoIdx]
    plo = tle_array[ploIdx]
    se2 = tle_array[se2Idx]
    se3 = tle_array[se3Idx]
    sgh2 = tle_array[sgh2Idx]
    sgh3 = tle_array[sgh3Idx]
    sgh4 = tle_array[sgh4Idx]
    sh2 = tle_array[sh2Idx]
    sh3 = tle_array[sh3Idx]
    si2 = tle_array[si2Idx]
    si3 = tle_array[si3Idx]
    sl2 = tle_array[sl2Idx]
    sl3 = tle_array[sl3Idx]
    sl4 = tle_array[sl4Idx]
    t = tle_array[tIdx]
    xgh2 = tle_array[xgh2Idx]
    xgh3 = tle_array[xgh3Idx]
    xgh4 = tle_array[xgh4Idx]
    xh2 = tle_array[xh2Idx]
    xh3 = tle_array[xh3Idx]
    xi2 = tle_array[xi2Idx]
    xi3 = tle_array[xi3Idx]
    xl2 = tle_array[xl2Idx]
    xl3 = tle_array[xl3Idx]
    xl4 = tle_array[xl4Idx]
    zmol = tle_array[zmolIdx]
    zmos = tle_array[zmosIdx]

    # Elements to be updated
    ep = tle_array[epIdx]
    inclp = tle_array[inclpIdx]
    nodep = tle_array[nodepIdx]
    argpp = tle_array[argppIdx]
    mp = tle_array[mpIdx]

    # Calculate time-varying periodics
    if init == 121:  # 'y' in ASCII
        zm = zmos
    else:
        zm = zmos + zns * t
    zf = zm + 2.0 * zes * math.sin(zm)
    sinzf = math.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * math.cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3

    if init == 121:
        zm = zmol
    else:
        zm = zmol + znl * t
    zf = zm + 2.0 * zel * math.sin(zm)
    sinzf = math.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * math.cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3

    # Total perturbations
    pe = ses + sel
    pinc = sis + sil
    pl = sls + sll
    pgh = sghs + sghl
    ph = shs + shll

    if init == 110:  # 'n' in ASCII
        # Apply secular effects
        pe = pe - peo
        pinc = pinc - pinco
        pl = pl - plo
        pgh = pgh - pgho
        ph = ph - pho
        ep = ep + pe
        inclp = inclp + pinc
        sinip = math.sin(inclp)
        cosip = math.cos(inclp)

        if inclp >= 0.2:
            # Apply periodics directly
            ph = ph / sinip
            pgh = pgh - cosip * ph
            argpp = argpp + pgh
            nodep = nodep + ph
            mp = mp + pl
        else:
            # Apply periodics with Lyddane's modification
            sinop = math.sin(nodep)
            cosop = math.cos(nodep)
            alfdp = sinip * sinop
            betdp = sinip * cosop
            dalf = ph * cosop + pinc * cosip * sinop
            dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet

            # Correct nodep
            nodep = math.fmod(nodep, twopi)
            if nodep < 0.0 and opsmode == 97.0:  # 'a' in ASCII, use float comparison
                nodep += twopi

            # Compute xls
            xls = mp + argpp + cosip * nodep
            dls = pl + pgh - pinc * nodep * sinip
            xls += dls
            xls = math.fmod(xls, twopi)

            # Save old nodep and update it
            xnoh = nodep
            nodep = math.atan2(alfdp, betdp)
            if nodep < 0.0 and opsmode == 97.0:  # 'a' in ASCII, use float comparison
                nodep += twopi

            # Adjust nodep to prevent discontinuity
            if math.fabs(xnoh - nodep) > math.pi:
                if nodep < xnoh:
                    nodep += twopi
                else:
                    nodep -= twopi

            # Update mean anomaly and argument of perigee
            mp += pl
            argpp = xls - mp - cosip * nodep

    # Save updated values back into tle_array
    tle_array[epIdx] = ep
    tle_array[inclpIdx] = inclp
    tle_array[nodepIdx] = nodep
    tle_array[argppIdx] = argpp
    tle_array[mpIdx] = mp


@cuda.jit(device=True)
def dspace(tle_array, tc):
    """Compute deep space resonance effects."""
    # Initialize variables
    xndt = 0.0
    xnddt = 0.0
    xldot = 0.0
    ft = 0.0

    # Constants as per original code
    fasx2 = 0.13130908
    fasx4 = 2.8843198
    fasx6 = 0.37448087
    g22 = 5.7686396
    g32 = 0.95240898
    g44 = 1.8014998
    g52 = 1.0508330
    g54 = 4.4108898
    rptim = 4.37526908801129966e-3  # rad/sec
    stepp = 720.0
    stepn = -720.0
    step2 = 259200.0

    # ------------------ Calculate Deep Space Resonance Effects ------------------

    # Initialize dndt
    tle_array[dndtIdx] = 0.0

    # Calculate theta with modulo operation
    theta = math.fmod(tle_array[gstoIdx] + tc * rptim, twopi)

    # Update em
    tle_array[emIdx] += tle_array[dedtIdx] * tle_array[tIdx]

    # Update inclm
    tle_array[inclmIdx] += tle_array[didtIdx] * tle_array[tIdx]

    # Update argpm
    tle_array[argpmIdx] += tle_array[domdtIdx] * tle_array[tIdx]

    # Update nodem
    tle_array[nodemIdx] += tle_array[dnodtIdx] * tle_array[tIdx]

    # Update mm
    tle_array[mmIdx] += tle_array[dmdtIdx] * tle_array[tIdx]

    # ------------------ Handle Resonances ------------------

    if tle_array[irezIdx] != 0:
        # Check if atime needs to be reset
        if (tle_array[atimeIdx] == 0.0) or (tle_array[tIdx] * tle_array[atimeIdx] <= 0.0) or (
                math.fabs(tle_array[tIdx]) < math.fabs(tle_array[atimeIdx])):
            tle_array[atimeIdx] = 0.0
            tle_array[xniIdx] = tle_array[no_unkozaiIdx]
            tle_array[xliIdx] = tle_array[xlamoIdx]

        # Determine integration step direction
        delt = stepp if tle_array[tIdx] > 0.0 else stepn

        iretn = 381  # Control variable for loop
        while iretn == 381:
            # Near-synchronous resonance terms
            if tle_array[irezIdx] != 2:
                xndt = (tle_array[del1Idx] * math.sin(tle_array[xliIdx] - fasx2) +
                        tle_array[del2Idx] * math.sin(2.0 * (tle_array[xliIdx] - fasx4)) +
                        tle_array[del3Idx] * math.sin(3.0 * (tle_array[xliIdx] - fasx6)))
                xldot = tle_array[xniIdx] + tle_array[xfactIdx]
                xnddt = (tle_array[del1Idx] * math.cos(tle_array[xliIdx] - fasx2) +
                         2.0 * tle_array[del2Idx] * math.cos(2.0 * (tle_array[xliIdx] - fasx4)) +
                         3.0 * tle_array[del3Idx] * math.cos(3.0 * (tle_array[xliIdx] - fasx6)))
                xnddt *= xldot
            else:
                # Near half-day resonance terms
                xomi = tle_array[argpoIdx] + tle_array[argpdotIdx] * tle_array[atimeIdx]
                x2omi = xomi + xomi
                x2li = tle_array[xliIdx] + tle_array[xliIdx]
                xndt = (tle_array[d2201Idx] * math.sin(x2omi + tle_array[xliIdx] - g22) +
                        tle_array[d2211Idx] * math.sin(tle_array[xliIdx] - g22) +
                        tle_array[d3210Idx] * math.sin(xomi + tle_array[xliIdx] - g32) +
                        tle_array[d3222Idx] * math.sin(-xomi + tle_array[xliIdx] - g32) +
                        tle_array[d4410Idx] * math.sin(x2omi + x2li - g44) +
                        tle_array[d4422Idx] * math.sin(x2li - g44) +
                        tle_array[d5220Idx] * math.sin(xomi + tle_array[xliIdx] - g52) +
                        tle_array[d5232Idx] * math.sin(-xomi + tle_array[xliIdx] - g52) +
                        tle_array[d5421Idx] * math.sin(xomi + x2li - g54) +
                        tle_array[d5433Idx] * math.sin(-xomi + x2li - g54))
                xldot = tle_array[xniIdx] + tle_array[xfactIdx]
                xnddt = (tle_array[d2201Idx] * math.cos(x2omi + tle_array[xliIdx] - g22) +
                         tle_array[d2211Idx] * math.cos(tle_array[xliIdx] - g22) +
                         tle_array[d3210Idx] * math.cos(xomi + tle_array[xliIdx] - g32) +
                         tle_array[d3222Idx] * math.cos(-xomi + tle_array[xliIdx] - g32) +
                         tle_array[d5220Idx] * math.cos(xomi + tle_array[xliIdx] - g52) +
                         tle_array[d5232Idx] * math.cos(-xomi + tle_array[xliIdx] - g52) +
                         2.0 * (tle_array[d4410Idx] * math.cos(x2omi + x2li - g44) +
                                tle_array[d4422Idx] * math.cos(x2li - g44) +
                                tle_array[d5421Idx] * math.cos(xomi + x2li - g54) +
                                tle_array[d5433Idx] * math.cos(-xomi + x2li - g54)))
                xnddt *= xldot

            # Integrator
            if math.fabs(tle_array[tIdx] - tle_array[atimeIdx]) >= stepp:
                iretn = 381
            else:
                ft = tle_array[tIdx] - tle_array[atimeIdx]
                iretn = 0

            if iretn == 381:
                # Update xli and xni
                xli_new = tle_array[xliIdx] + xldot * delt + xndt * step2
                xni_new = tle_array[xniIdx] + xndt * delt + xnddt * step2
                # Update tle_array
                tle_array[xliIdx] = xli_new
                tle_array[xniIdx] = xni_new
                # Update atime
                tle_array[atimeIdx] += delt

        # Update nm and mm
        nm = tle_array[xniIdx] + xndt * ft + xnddt * ft * ft * 0.5
        xl = tle_array[xliIdx] + xldot * ft + xndt * ft * ft * 0.5

        if tle_array[irezIdx] != 1:
            mm = xl - 2.0 * tle_array[nodemIdx] + 2.0 * theta
            dndt = nm - tle_array[no_unkozaiIdx]
        else:
            mm = xl - tle_array[nodemIdx] - tle_array[argpmIdx] + theta
            dndt = nm - tle_array[no_unkozaiIdx]

        nm = tle_array[no_unkozaiIdx] + dndt

        # Save updated values back into tle_array
        tle_array[nmIdx] = nm
        tle_array[mmIdx] = mm
        tle_array[dndtIdx] = dndt


@cuda.jit(device=True, fastmath=False)
def sgp4(tle_array, tsince, r, v):
    """Propagate a single satellite for ``tsince`` minutes."""
    # Define mathematical constants
    x2o3 = 2.0 / 3.0
    temp4 = 1.5e-12
    twopi = 2.0 * math.pi
    pi = math.pi

    # Extract necessary variables from tle_array and cast to appropriate types
    nm = tle_array[nmIdx]
    em = tle_array[emIdx]
    inclm = tle_array[inclmIdx]
    nodem = tle_array[nodemIdx]
    argpm = tle_array[argpmIdx]
    mm = tle_array[mmIdx]
    no_unkozai = tle_array[no_unkozaiIdx]
    ecco = tle_array[eccoIdx]
    inclo = tle_array[incloIdx]
    nodeo = tle_array[nodeoIdx]
    argpo = tle_array[argpoIdx]
    sinim = tle_array[sinimIdx]
    cosim = tle_array[cosimIdx]
    sinomm = tle_array[sinommIdx]
    cosomm = tle_array[cosommIdx]
    snodm = tle_array[snodmIdx]
    cnodm = tle_array[cnodmIdx]
    emsq = tle_array[emsqIdx]
    rtemsq = tle_array[rtemsqIdx]
    isimp = int32(tle_array[isimpIdx])  # Cast to integer
    omgcof = tle_array[omgcofIdx]
    xmcof = tle_array[xmcofIdx]
    delmo = tle_array[delmoIdx]
    d2 = tle_array[d2Idx]
    d3 = tle_array[d3Idx]
    d4 = tle_array[d4Idx]
    bstar = tle_array[bstarIdx]
    cc1 = tle_array[cc1Idx]
    cc4 = tle_array[cc4Idx]
    cc5 = tle_array[cc5Idx]
    t2cof = tle_array[t2cofIdx]
    t3cof = tle_array[t3cofIdx]
    t4cof = tle_array[t4cofIdx]
    t5cof = tle_array[t5cofIdx]
    xke = tle_array[xkeIdx]
    radiusearthkm = tle_array[radiusearthkmIdx]
    j2oj2 = tle_array[j3oj2Idx]  # Assuming j3oj2 is defined correctly
    method = int32(tle_array[methodIdx])  # Cast to integer
    operationmode = int32(tle_array[operationmodeIdx])  # Cast to integer
    # ... extract any other necessary variables

    # Initialize variables
    tle_array[tIdx] = tsince
    tle_array[errorIdx] = 0  # Clear error flag

    # Update for secular gravity and atmospheric drag
    xmdf = tle_array[moIdx] + tle_array[mdotIdx] * tle_array[tIdx]
    argpdf = argpo + tle_array[argpdotIdx] * tle_array[tIdx]
    nodedf = nodeo + tle_array[nodedotIdx] * tle_array[tIdx]
    tle_array[argpmIdx] = argpdf
    tle_array[mmIdx] = xmdf
    t2 = tle_array[tIdx] * tle_array[tIdx]
    nodem = nodedf + tle_array[nodecfIdx] * t2
    tle_array[nodemIdx] = nodem
    tempa = 1.0 - cc1 * tle_array[tIdx]
    tempe = bstar * cc4 * tle_array[tIdx]
    templ = t2cof * t2

    # Initialize temporary variables
    delomg = 0.0
    delmtemp = 0.0
    delm = 0.0
    temp = 0.0
    t3 = 0.0
    t4 = 0.0
    mrt = 0.0

    if isimp != 1:
        delomg = omgcof * tle_array[tIdx]
        delmtemp = 1.0 + tle_array[etaIdx] * math.cos(xmdf)
        delm = xmcof * (delmtemp * delmtemp * delmtemp - delmo)
        temp = delomg + delm
        tle_array[mmIdx] = xmdf + temp
        tle_array[argpmIdx] = argpdf - temp
        argpm = tle_array[argpmIdx]
        t3 = t2 * tle_array[tIdx]
        t4 = t3 * tle_array[tIdx]
        tempa = tempa - d2 * t2 - d3 * t3 - d4 * t4
        tempe = tempe + bstar * cc5 * (math.sin(tle_array[mmIdx]) - tle_array[sinmaoIdx])
        templ = templ + t3cof * t3 + t4 * (tle_array[t4cofIdx] + tle_array[t5cofIdx] * tle_array[tIdx])

    tc = 0.0
    nm = no_unkozai
    tle_array[nmIdx] = nm
    tle_array[emIdx] = ecco
    tle_array[inclmIdx] = inclo

    if method == ord('d'):  # Assuming method is stored as ASCII code
        tc = tle_array[tIdx]
        dspace(tle_array, tc)  # Adapted dspace function should be used

    nm = tle_array[nmIdx]
    em = tle_array[emIdx]
    mm = tle_array[mmIdx]
    inclm = tle_array[inclmIdx]
    argpm = tle_array[argpmIdx]
    nodem = tle_array[nodemIdx]

    # Check for valid mean motion
    if nm <= 0.0:
        tle_array[errorIdx] = 2
        return  # Exit the function early

    am = math.pow((xke / nm), x2o3) * tempa * tempa
    tle_array[amIdx] = am
    nm = xke / math.pow(am, 1.5)
    tle_array[nmIdx] = nm
    em = em - tempe
    tle_array[emIdx] = em

    # Fix tolerance for error recognition
    if em >= 1.0 or em < -0.001:
        tle_array[errorIdx] = 1
        return  # Exit the function early

    # Fix tolerance to avoid a divide by zero
    if em < 1.0e-6:
        em = 1.0e-6
        tle_array[emIdx] = em

    tle_array[mmIdx] = tle_array[mmIdx] + no_unkozai * templ
    xlm = tle_array[mmIdx] + argpm + nodem
    tle_array[emsqIdx] = em * em
    temp = 1.0 - tle_array[emsqIdx]

    # Normalize angles using modulo operator
    tle_array[nodemIdx] = math.fmod(nodem, twopi)
    tle_array[argpmIdx] = math.fmod(argpm, twopi)
    xlm = math.fmod(xlm, twopi)
    tle_array[mmIdx] = math.fmod(xlm - tle_array[argpmIdx] - tle_array[nodemIdx], twopi)

    # Recover singly averaged mean elements
    tle_array[amIdx] = tle_array[amIdx]
    tle_array[emIdx] = tle_array[emIdx]
    tle_array[imIdx] = tle_array[inclmIdx]
    tle_array[OmIdx] = tle_array[nodemIdx]
    tle_array[omIdx] = tle_array[argpmIdx]
    tle_array[mmIdx] = tle_array[mmIdx]
    tle_array[nmIdx] = tle_array[nmIdx]

    # Compute extra mean quantities
    sinim = math.sin(inclm)
    cosim = math.cos(inclm)
    tle_array[sinimIdx] = sinim
    tle_array[cosimIdx] = cosim

    # Add lunar-solar periodics
    ep = em
    tle_array[epIdx] = ep
    xincp = inclm
    inclp = inclm
    tle_array[inclpIdx] = inclp
    argpp = tle_array[argpmIdx]
    tle_array[argppIdx] = argpp
    nodep = nodem
    tle_array[nodepIdx] = nodep
    mp = tle_array[mmIdx]
    tle_array[mpIdx] = mp
    sinip = sinim
    cosip = cosim

    if method == ord('d'):
        dpper(tle_array, ord('n'), operationmode)  # Ensure operationmode is correctly passed
        ep = tle_array[epIdx]
        inclp = tle_array[inclpIdx]
        nodep = tle_array[nodepIdx]
        argpp = tle_array[argppIdx]
        mp = tle_array[mpIdx]

        xincp = tle_array[inclpIdx]
        if xincp < 0.0:
            xincp = -xincp
            tle_array[nodepIdx] = tle_array[nodepIdx] + pi
            tle_array[argppIdx] = tle_array[argppIdx] - pi
            # Update local variables to match the array values
            nodep = tle_array[nodepIdx]
            argpp = tle_array[argppIdx]

        if ep < 0.0 or ep > 1.0:
            tle_array[errorIdx] = 3
            return  # Exit the function early

    # Long period periodics
    if method == ord('d'):
        sinip = math.sin(xincp)
        cosip = math.cos(xincp)
        tle_array[aycofIdx] = -0.5 * j2oj2 * sinip
        # Fix for divide by zero for xincp = 180 deg
        if abs(cosip + 1.0) > 1.5e-12:
            tle_array[xlcofIdx] = -0.25 * j2oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip)
        else:
            tle_array[xlcofIdx] = -0.25 * j2oj2 * sinip * (3.0 + 5.0 * cosip) / temp4

    # Compute orbital elements
    axnl = tle_array[epIdx] * math.cos(tle_array[argppIdx])
    temp = 1.0 / (tle_array[amIdx] * (1.0 - tle_array[epIdx] * tle_array[epIdx]))
    aynl = tle_array[epIdx] * math.sin(tle_array[argppIdx]) + temp * tle_array[aycofIdx]
    xl = mp + tle_array[argppIdx] + tle_array[nodepIdx] + temp * tle_array[xlcofIdx] * axnl
    # Solve Kepler's equation
    u = math.fmod(xl - nodep, twopi)
    eo1 = u
    tem5 = 9999.9
    ktr = 1
    sineo1 = 0.0
    coseo1 = 0.0

    while abs(tem5) >= 1.0e-12 and ktr <= 10:
        sineo1 = math.sin(eo1)
        coseo1 = math.cos(eo1)
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        if abs(tem5) >= 0.95:
            tem5 = 0.95 * (1 if tem5 > 0 else -1)
        eo1 = eo1 + tem5
        ktr += 1

    # Short period preliminary quantities
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2 = axnl * axnl + aynl * aynl
    pl = tle_array[amIdx] * (1.0 - el2)

    if pl < 0.0:
        tle_array[errorIdx] = 4
        return  # Exit the function early
    else:
        rl = tle_array[amIdx] * (1.0 - ecose)
        rdotl = math.sqrt(tle_array[amIdx]) * esine / rl
        rvdotl = math.sqrt(pl) / rl
        betal = math.sqrt(1.0 - el2)
        temp = esine / (1.0 + betal)
        sinu = tle_array[amIdx] / rl * (sineo1 - aynl - axnl * temp)
        cosu = tle_array[amIdx] / rl * (coseo1 - axnl + aynl * temp)
        su = math.atan2(sinu, cosu)
        sin2u = 2.0 * sinu * cosu
        cos2u = 1.0 - 2.0 * sinu * sinu
        temp = 1.0 / pl
        temp1 = 0.5 * tle_array[j2Idx] * temp
        temp2 = temp1 * temp

        if method == ord('d'):
            cosisq = cosip * cosip
            tle_array[con41Idx] = 3.0 * cosisq - 1.0
            tle_array[x1mth2Idx] = 1.0 - cosisq
            tle_array[x7thm1Idx] = 7.0 * cosisq - 1.0

        mrt = rl * (1.0 - 1.5 * temp2 * betal * tle_array[con41Idx]) + 0.5 * temp1 * tle_array[x1mth2Idx] * cos2u
        su = su - 0.25 * temp2 * tle_array[x7thm1Idx] * sin2u
        xnode = nodep + 1.5 * temp2 * cosip * sin2u
        xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
        mvt = rdotl - nm * temp1 * tle_array[x1mth2Idx] * sin2u / xke
        rvdot = rvdotl + nm * temp1 * (tle_array[x1mth2Idx] * cos2u + 1.5 * tle_array[con41Idx]) / xke

        # Orientation vectors
        sinsu = math.sin(su)
        cossu = math.cos(su)
        snod = math.sin(xnode)
        cnod = math.cos(xnode)
        sini = math.sin(xinc)
        cosi = math.cos(xinc)
        xmx = -snod * cosi
        xmy = cnod * cosi
        ux = xmx * sinsu + cnod * cossu
        uy = xmy * sinsu + snod * cossu
        uz = sini * sinsu
        vx = xmx * cossu - cnod * sinsu
        vy = xmy * cossu - snod * sinsu
        vz = sini * cossu

        # Position and velocity (in km and km/sec)
        r[0] = mrt * ux * radiusearthkm
        r[1] = mrt * uy * radiusearthkm
        r[2] = mrt * uz * radiusearthkm
        vkmpersec = radiusearthkm * xke / 60.0  # Assuming this calculation is correct
        v[0] = (mvt * ux + rvdot * vx) * vkmpersec
        v[1] = (mvt * uy + rvdot * vy) * vkmpersec
        v[2] = (mvt * uz + rvdot * vz) * vkmpersec

        # Fix for decaying satellites
        if mrt < 1.0:
            tle_array[errorIdx] = 6
            return False  # Exit the function early

    # Function completed successfully
    return True

@cuda.jit
def propagate_orbit(tle_arrays, r, v, total_timesteps, timestep_length_in_seconds):
    """Kernel wrapper calling :func:`sgp4` for each satellite and timestep."""
    idx = cuda.grid(1)
    num_satellites = tle_arrays.shape[0]

    if idx >= num_satellites:
        return  # Exit if thread index exceeds number of satellites

    # Extract the satellite's tle_array
    tle_array = tle_arrays[idx, :]

    time_diff = tle_array[tIdx]

    # Iterate over each timestep
    for t_step in range(total_timesteps):
        tsince = time_diff + t_step * timestep_length_in_seconds / 60.0  # Convert to minutes
        # Call the sgp4 device function to propagate
        sgp4(tle_array, tsince, r[:, idx, t_step], v[:, idx, t_step])

@cuda.jit
def _transpose_arrays(d_r, d_v, d_r_transposed, d_v_transposed, num_satellites, total_timesteps):
    """Transpose position and velocity arrays from (3, n_sats, n_steps) to (n_sats, n_steps, 3) on GPU."""
    idx = cuda.grid(1)
    
    if idx >= num_satellites:
        return
    
    for t_step in range(total_timesteps):
        # Transpose positions: (3, n_sats, n_steps) -> (n_sats, n_steps, 3)
        d_r_transposed[idx, t_step, 0] = d_r[0, idx, t_step]
        d_r_transposed[idx, t_step, 1] = d_r[1, idx, t_step]
        d_r_transposed[idx, t_step, 2] = d_r[2, idx, t_step]
        
        # Transpose velocities: (3, n_sats, n_steps) -> (n_sats, n_steps, 3)
        d_v_transposed[idx, t_step, 0] = d_v[0, idx, t_step]
        d_v_transposed[idx, t_step, 1] = d_v[1, idx, t_step]
        d_v_transposed[idx, t_step, 2] = d_v[2, idx, t_step]
