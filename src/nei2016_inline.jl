export NEI2016InlineEmis

# ============================================================================
# NEI 2016 INLINE elevated point-source emissions (EGU power plants, oil & gas,
# large fires, C3 marine, ...).
#
# Counterpart to `NEI2016MonthlyEmis` (the 2-D surface-gridded merge), addressing
# EarthSciML/EarthSciData.jl#211: the surface merge `mrggrid_withbeis_withrwc`
# EXCLUDES every "inline-only" elevated point sector. `ptegu` alone (EGU power
# plants) is ~2.2 of the ~2.5 Tg/yr CONUS SO2 budget and contributes ZERO to the
# layer-1 surface merge. Those emissions exist only as CMAQ INLINE point sources:
#
#   merged/12US1_inln/2016fh_12US1_<sector>_inln.zip
#
# which bundle, on the 12US1 Lambert-Conformal grid (459 col x 299 row):
#   * stack_groups_<sector>_..._16j.ncf  — one record per stack (LATITUDE,
#     LONGITUDE, STKHT, STKDM, STKTK, STKVE, STKFLW, ROW, COL, ...), time-invariant
#     geometry. NETCDF3_64BIT_OFFSET; stacks are stored along the ROW dimension
#     (ROW = NSTACKS, COL = 1).
#   * inln_mole_<sector>_YYYYMMDD_..._16j.ncf — per representative day, hourly
#     (TSTEP = 25), per-species emission rate in **mole/s**, indexed by the SAME
#     stack ordering (EMIS dims TSTEP x LAY(=1) x ROW(=NSTACKS) x COL(=1)).
#
# DESIGN (the seam that lets point sources reuse the gridded machinery):
#  - HORIZONTAL placement is STATIC: each stack maps to a fixed 12US1 grid cell
#    (its integer ROW/COL). `loadslice!` SCATTERS the per-stack emissions into a
#    dense 12US1 grid array, so the EXISTING conservative regridder (source-cell
#    polygon -> model-cell polygon) carries it to the simulation grid unchanged.
#    After the scatter this FileSet is an ordinary gridded 12US1 source, so
#    `loadmetadata` / `get_geometry` mirror `NEI2016MonthlyEmis` (same grid).
#  - VERTICAL placement is DYNAMIC (plume rise depends on the meteorological state
#    at solve time) and is therefore NOT done in the FileSet. It is applied in the
#    System builder's per-cell `wrapper_f` (`NEI2016InlineEmis`, added separately)
#    using gridded stack parameters + GEOSFP meteorology (Z_agl, T, wind), i.e. a
#    state-dependent generalization of the surface loader's `ifelse(lev<2, …)`.
#  - mole/s -> kg/s via per-species molecular weights (`_INLINE_MW`); the inline
#    files are mole speciation, unlike the mass (`tons/day`) surface merge.
#  - The #209 days-in-month fix and the synthetic diurnal/day-of-week scaling of
#    `NEI2016MonthlyEmis` DO NOT apply here: the inline files are already true
#    hourly rates, so neither correction is used (applying them would double-count).
#
# ⚠ STATUS: Phase 1 = this FileSet (geometry + emission scatter -> 12US1 grid).
#    The plume-rise vertical wrapper + the `NEI2016InlineEmis` System builder are
#    added in a following step. Parts that must be exercised against real data /
#    a live Julia session before trusting numbers are marked `# VERIFY`.
# ============================================================================

# Molecular weights [g/mol] for converting the mole-speciation inline emissions
# to mass. Covers the GEOS-Chem-relevant CB6 inline species; lumped CB6 species
# (PAR, OLE, ...) use the CMAQ-convention carbon/representative weights. Extend as
# needed — `loadslice!` warns (once) for any emitted species missing here.
# VERIFY: confirm the exact inline species set + units string against a real
# inln_mole header (dims already confirmed: TSTEP x LAY x ROW(=NSTACKS) x COL).
const _INLINE_MW = Dict{String, Float64}(
    "NO" => 30.006, "NO2" => 46.006, "HONO" => 47.013, "SO2" => 64.066,
    "SULF" => 98.079, "CO" => 28.010, "NH3" => 17.031, "CO2" => 44.009,
    "FORM" => 30.026, "FORM_PRIMARY" => 30.026, "ALD2" => 44.053,
    "ALD2_PRIMARY" => 44.053, "ALDX" => 58.080, "ACET" => 58.080,
    "ETH" => 28.054, "ETHA" => 30.070, "ETHY" => 26.038, "ETOH" => 46.069,
    "MEOH" => 32.042, "PAR" => 14.043, "OLE" => 27.000, "IOLE" => 56.108,
    "ISOP" => 68.117, "TERP" => 136.234, "BENZ" => 78.114, "TOL" => 92.140,
    "XYLMN" => 106.165, "NAPH" => 128.171, "PRPA" => 44.096, "KET" => 72.000,
    "CH4" => 16.043, "ACROLEIN" => 56.064, "BUTADIENE13" => 54.092,
    "CL2" => 70.906, "HCL" => 36.461, "GLY" => 58.036, "GLYD" => 60.052,
    "MGLY" => 72.063,
)

# ----------------------------------------------------------------------------
# StackTable: per-stack geometry, read once from the stack_groups file. Holds the
# horizontal cell index (col/row into the 12US1 grid) used by `loadslice!`'s
# scatter, and the stack parameters used later for plume rise.
# ----------------------------------------------------------------------------
struct StackTable
    n::Int
    lon::Vector{Float64}    # degrees
    lat::Vector{Float64}    # degrees
    col::Vector{Int}        # 1-based 12US1 grid column index
    row::Vector{Int}        # 1-based 12US1 grid row index
    stkht::Vector{Float64}  # stack height above ground [m]
    stkdm::Vector{Float64}  # inside stack diameter [m]
    stktk::Vector{Float64}  # exit temperature [K]
    stkve::Vector{Float64}  # exit velocity [m/s]
end

# Read the (TSTEP=1, LAY=1, ROW=NSTACKS, COL=1) stack-groups variables into flat
# per-stack vectors. NCDatasets presents the IOAPI dims reversed (COL, ROW, LAY,
# TSTEP); each geometry variable is singleton in every dim except ROW, so `vec`
# yields the NSTACKS-length vector.
function _read_stack_groups(ds)
    g(name) = Float64.(vec(Array(ds[name])))
    lon = g("LONGITUDE")
    n = length(lon)
    return StackTable(
        n, lon, g("LATITUDE"),
        round.(Int, g("COL")), round.(Int, g("ROW")),
        g("STKHT"), g("STKDM"), g("STKTK"), g("STKVE"),
    )
end

# ----------------------------------------------------------------------------
# FileSet
# ----------------------------------------------------------------------------
"""
$(SIGNATURES)

`FileSet` for CMAQ inline (elevated) NEI 2016 point-source emissions for one
sector. See the file header for the data layout and design. Parameterized on the
inline-emissions dataset type `D` for type-stable NetCDF reads.
"""
struct NEI2016InlineEmisFileSet{D} <: FileSet
    mirror::String
    sector::String
    stacks::StackTable
    ds::D                       # aggregated inln_mole dataset (representative days)
    freq_info::DataFrequencyInfo
    # 12US1 target grid + LCC projection (from the stack_groups global attrs;
    # NCOLS/NROWS are the full grid, GDNAM "12US1_459X299").
    ncols::Int
    nrows::Int
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
    native_sr::String
end

DataFrequencyInfo(fs::NEI2016InlineEmisFileSet) = fs.freq_info
Base.close(fs::NEI2016InlineEmisFileSet) = lock(nclock) do; close(fs.ds); end

"""
$(SIGNATURES)

Server path (relative to the host root / local cache) of the inline zip archive
for this sector. Sector-agnostic: the same layout holds for every inline sector
(ptegu, ptnonipm, pt_oilgas, cmv_c3, ptfire, ptagfire, ...).
"""
function relpath(fs::NEI2016InlineEmisFileSet, t::DateTime)
    @assert Dates.year(t)==2016 "Only 2016 emissions are available with `NEI2016InlineEmis`."
    return "emismod/2016/v1/merged/12US1_inln/2016fh_12US1_$(fs.sector)_inln.zip"
end

"""
$(SIGNATURES)

Build the metadata describing the 12US1 grid that `loadslice!` scatters the
point emissions onto. Identical in spirit to `NEI2016MonthlyEmis` — the scattered
field IS a 12US1 gridded field — so the shared conservative regridder applies.
The species emission units after scatter + mole->mass are kg/m²/s.
"""
function loadmetadata(fs::NEI2016InlineEmisFileSet, varname)::MetaData
    xs = fs.x0 + fs.dx / 2 .+ fs.dx .* (0:(fs.ncols - 1))
    ys = fs.y0 + fs.dy / 2 .+ fs.dy .* (0:(fs.nrows - 1))
    return MetaData(
        [xs, ys],
        "kg/m^2/s",
        "NEI 2016 inline point-source emissions of $(varname) (scattered to 12US1)",
        ["COL", "ROW"],
        [fs.ncols, fs.nrows],
        fs.native_sr,
        1,        # xdim (COL)
        2,        # ydim (ROW)
        -1,       # zdim (none — vertical placement is in the System wrapper)
        (false, false, false),
    )
end

"""
$(SIGNATURES)

12US1 source-cell polygons for conservative regridding (column-major, x-fastest,
matching `vec` on the scattered data array). Mirrors `NEI2016MonthlyEmis`.
"""
function get_geometry(fs::NEI2016InlineEmisFileSet, m::MetaData)
    x = range(start = fs.x0, step = fs.dx, length = fs.ncols + 1)
    y = range(start = fs.y0, step = fs.dy, length = fs.nrows + 1)
    polys = Vector{Vector{NTuple{2, Float64}}}(undef, fs.ncols * fs.nrows)
    for j in 1:(fs.nrows), i in 1:(fs.ncols)
        polys[(j - 1) * fs.ncols + i] = [(x[i], y[j]), (x[i + 1], y[j]),
            (x[i + 1], y[j + 1]), (x[i], y[j + 1]), (x[i], y[j])]
    end
    return polys
end

"""
$(SIGNATURES)

Species variable names in the inline file (excludes TFLAG and dimension vars).
"""
function varnames(fs::NEI2016InlineEmisFileSet)
    lock(nclock) do
        return [setdiff(keys(fs.ds), ["TFLAG"; keys(fs.ds.dim)])...]
    end
end

"""
$(SIGNATURES)

Load one species' inline emissions at time `t` and SCATTER it onto the 12US1
grid: read the per-stack mole/s vector, convert to kg/s with the species
molecular weight, accumulate each stack into its (col,row) cell, then divide by
cell area to get the kg/m²/s flux density the conservative regridder expects.
"""
function loadslice!(data::AbstractArray, fs::NEI2016InlineEmisFileSet, t::DateTime, varname)
    mw = get(_INLINE_MW, String(varname), NaN)
    isnan(mw) && (@warn "No molecular weight for inline species $(varname); emitting 0." maxlog=1)
    lock(nclock) do
        fill!(data, 0)
        ti = _inline_time_index(fs, t)              # representative-day + hour index
        var = fs.ds[varname]
        # EMIS dims (NCDatasets order): COL(=1) x ROW(=NSTACKS) x LAY(=1) x TSTEP.
        emis = Float64.(vec(Array(var[:, :, :, ti])))   # mole/s per stack
        st = fs.stacks
        @assert length(emis)==st.n "inline EMIS length $(length(emis)) != NSTACKS $(st.n)"
        if !isnan(mw)
            mols2kgs = mw * 1.0e-3                   # mole/s * g/mol * 1e-3 kg/g = kg/s
            @inbounds for s in 1:st.n
                c = st.col[s]; r = st.row[s]
                (1 <= c <= fs.ncols && 1 <= r <= fs.nrows) || continue
                data[c, r] += emis[s] * mols2kgs     # kg/s accumulated into the cell
            end
        end
        data ./= (fs.dx * fs.dy)                     # kg/s -> kg/m²/s
    end
    nothing
end

# 1-based time index into the aggregated inline dataset for model time `t`.
# VERIFY: representative-day mapping. EPA inline files cover representative days
# (e.g. a weekday/Sat/Sun per month), not every calendar day. This first cut
# selects the nearest available centerpoint hour; the precise EPA representative-
# day scheme (day-type per month) is refined once the aggregated TFLAG set is
# inspected on real data.
function _inline_time_index(fs::NEI2016InlineEmisFileSet, t::DateTime)
    cps = fs.freq_info.centerpoints
    return argmin(abs.(Dates.value.(cps .- t)))
end

# ----------------------------------------------------------------------------
# Construction: fetch + open the inline archive for the sector.
#
# The inline zips are large (multi-GB; pt_oilgas ~4.7 GB) and bundle one
# stack_groups file + many representative-day inln_mole files. Two access modes:
#   (a) cached extracted files under `download_cache()` (preferred for repeat
#       runs and for tests that pre-stage a small fixture);
#   (b) download + extract the needed members.
# VERIFY / TODO: for the multi-GB bundles, extract only the stack_groups (~2 MB)
# and the representative-day inln_mole files actually needed, rather than the
# whole archive (a ZIP64 central-directory + per-member range extraction was
# validated out-of-band and is the intended production path; the EDGAR-style
# whole-archive extract below is the simple correct fallback).
# ----------------------------------------------------------------------------
function NEI2016InlineEmisFileSet(sector::AbstractString, starttime::DateTime, endtime::DateTime;
        mirror = "https://gaftp.epa.gov/Air/")
    extract_dir = joinpath(download_cache(), "nei2016_inline", String(sector))

    stk_path = _inline_find(extract_dir, "stack_groups_$(sector)")
    inln_paths = _inline_find_all(extract_dir, "inln_mole_$(sector)_")
    if isnothing(stk_path) || isempty(inln_paths)
        _inline_ensure_extracted(mirror, sector, starttime, endtime, extract_dir)
        stk_path = _inline_find(extract_dir, "stack_groups_$(sector)")
        inln_paths = _inline_find_all(extract_dir, "inln_mole_$(sector)_")
    end
    @assert !isnothing(stk_path) "stack_groups file for sector $(sector) not found in $(extract_dir)"
    @assert !isempty(inln_paths) "no inln_mole files for sector $(sector) found in $(extract_dir)"

    lock(nclock) do
        stkds = NCDataset(stk_path)
        stacks = _read_stack_groups(stkds)
        # 12US1 grid + LCC projection from the stack_groups global attributes.
        a = stkds.attrib
        p_alp = a["P_ALP"]; p_bet = a["P_BET"]; xcent = a["XCENT"]; ycent = a["YCENT"]
        native_sr = "+proj=lcc +lat_1=$(p_alp) +lat_2=$(p_bet) +lat_0=$(ycent) " *
                    "+lon_0=$(xcent) +x_0=0 +y_0=0 +a=6370997.0 +b=6370997.0 +to_meter=1"
        x0 = a["XORIG"]; y0 = a["YORIG"]; dx = a["XCELL"]; dy = a["YCELL"]
        # NCOLS/NROWS in the sparse stack_groups are (1, NSTACKS); the full target
        # grid size comes from GDNAM ("12US1_459X299") -> 459 x 299.
        ncols, nrows = _parse_gdnam_size(get(a, "GDNAM", "12US1_459X299"))
        close(stkds)

        ds = NCDataset(sort(inln_paths), aggdim = "TSTEP")
        cps = sort(_inline_centerpoints(ds))
        start = DateTime(Dates.year(starttime), Dates.month(starttime))
        dfi = DataFrequencyInfo(start, Hour(1), cps)

        NEI2016InlineEmisFileSet{typeof(ds)}(String(mirror), String(sector), stacks,
            ds, dfi, ncols, nrows, x0, y0, dx, dy, native_sr)
    end
end

# Parse "12US1_459X299" -> (459, 299). Falls back to the standard 12US1 size.
function _parse_gdnam_size(gdnam)
    m = match(r"_(\d+)X(\d+)", strip(String(gdnam)))
    isnothing(m) ? (459, 299) : (parse(Int, m[1]), parse(Int, m[2]))
end

# Hourly centerpoints from the inline dataset's TFLAG (YYYYDDD, HHMMSS).
# VERIFY against the real aggregated TFLAG.
function _inline_centerpoints(ds)
    tf = Array(ds["TFLAG"])           # (DATE-TIME=2, VAR, TSTEP) in NCDatasets order
    nt = size(tf)[end]
    out = DateTime[]
    for k in 1:nt
        yyyyddd = Int(tf[1, 1, k]); hhmmss = Int(tf[2, 1, k])
        yr = yyyyddd ÷ 1000; doy = yyyyddd % 1000
        hh = hhmmss ÷ 10000
        push!(out, DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hh))
    end
    return out
end

function _inline_find(dir, prefix)
    isdir(dir) || return nothing
    for f in readdir(dir, join = true)
        startswith(basename(f), prefix) && endswith(f, ".ncf") && return f
    end
    return nothing
end

function _inline_find_all(dir, prefix)
    isdir(dir) || return String[]
    return filter(f -> startswith(basename(f), prefix) && endswith(f, ".ncf"),
        readdir(dir, join = true))
end

# Download the sector's inline zip and extract its .ncf members (EDGAR idiom).
# VERIFY / TODO: replace with range-based extraction of only the needed members
# for the multi-GB bundles.
function _inline_ensure_extracted(mirror, sector, starttime, endtime, extract_dir)
    rel = "emismod/2016/v1/merged/12US1_inln/2016fh_12US1_$(sector)_inln.zip"
    zip_url = rstrip(mirror, '/') * "/" * rel
    zip_local = joinpath(download_cache(), "nei2016_inline", "$(sector).zip")
    if !isfile(zip_local)
        mkpath(dirname(zip_local))
        @info "Downloading NEI inline archive (large) from $zip_url"
        _download_with_progress(zip_url, zip_local)
    end
    mkpath(extract_dir)
    @info "Extracting NEI inline .ncf members to $extract_dir"
    r = ZipFile.Reader(zip_local)
    try
        for f in r.files
            fn = basename(f.name)
            (endswith(fn, ".ncf") && !startswith(fn, ".")) || continue
            out = joinpath(extract_dir, fn)
            isfile(out) || open(out, "w") do io; write(io, read(f)); end
        end
    finally
        close(r)
    end
end
