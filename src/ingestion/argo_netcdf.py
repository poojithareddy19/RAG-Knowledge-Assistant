"""Read ARGO profile NetCDF files into rows for the measurements schema.

NetCDF is how ARGO is actually distributed, so this is the loader that reads
the archive in its native form rather than a CSV someone exported first.

Three things the file format forces on us, all handled here so that callers
only ever see plain values:

**Adjusted versus raw.** Every profile carries ``DATA_MODE``. In delayed mode
(``D``) and adjusted mode (``A``) the scientifically useful numbers are in the
``*_ADJUSTED`` variables, and the raw ones are kept only for provenance. In
real-time mode (``R``) the adjusted variables are empty. Reading the raw
variable in every mode is the single most common way to get ARGO wrong, so the
mode decides which pair is read.

**Per-parameter QC.** A level can have good temperature and bad salinity, so
each parameter carries its own flag. All three are kept, plus ``qc_flag``, the
worst of the flags whose parameter actually has a value. A reader who only
wants clean rows filters on ``qc_flag``; a reader who wants every usable
temperature filters on ``temperature_qc`` and ignores salinity.

**Char arrays.** Text is stored as arrays of single bytes, and times as days
since 1950. Both are decoded here.

This module does no database work, so it can be tested against a file alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.ingestion.regions import region_for

ARGO_EPOCH = pd.Timestamp("1950-01-01", tz="UTC")

# ARGO QC flags ordered by how much they should stop you using the value.
# Numeric order will not do: 8 (interpolated) is usable, 4 (bad) is not.
_QC_BADNESS = {1: 0, 2: 1, 5: 2, 8: 3, 3: 4, 4: 5, 9: 6}

# A position or time the float itself reports as bad. The region of a profile
# is derived from its position, so a bad fix poisons more than one column.
_REJECTED = 4

PARAMETERS = ("PRES", "TEMP", "PSAL")


@dataclass
class ArgoLevel:
    """One depth level within a profile."""

    pressure_dbar: float
    temperature_c: float | None
    salinity_psu: float | None
    pressure_qc: int | None
    temperature_qc: int | None
    salinity_qc: int | None
    qc_flag: int | None


@dataclass
class ArgoProfile:
    """One surfacing event, with every level it recorded."""

    float_id: int
    cycle_number: int
    obs_time: pd.Timestamp
    latitude: float
    longitude: float
    region: str
    data_mode: str
    platform: str
    project: str
    levels: list[ArgoLevel] = field(default_factory=list)


def read_file(path: str | Path) -> list[ArgoProfile]:
    """Read every usable profile out of one ARGO NetCDF file."""
    with xr.open_dataset(
        path,
        decode_times=False,
        mask_and_scale=True,
    ) as dataset:
        return _profiles(dataset)


def read_paths(paths) -> list[ArgoProfile]:
    """Read a mix of files and directories, recursing for ``*.nc``."""
    out = []

    for path in _expand(paths):
        out.extend(read_file(path))

    return out


def _expand(paths) -> list[Path]:
    found = []

    for raw in paths:
        path = Path(raw)

        if path.is_dir():
            found.extend(sorted(path.rglob("*.nc")))
        else:
            found.append(path)

    return found


def _profiles(dataset) -> list[ArgoProfile]:
    n_prof = dataset.sizes.get("N_PROF")

    if not n_prof:
        return []

    platforms = _strings(dataset, "PLATFORM_NUMBER", n_prof)
    projects = _strings(dataset, "PROJECT_NAME", n_prof)
    types = _strings(dataset, "PLATFORM_TYPE", n_prof)
    modes = _strings(dataset, "DATA_MODE", n_prof)
    position_qc = _strings(dataset, "POSITION_QC", n_prof)
    time_qc = _strings(dataset, "JULD_QC", n_prof)

    cycles = _numbers(dataset, "CYCLE_NUMBER", n_prof)
    juld = _numbers(dataset, "JULD", n_prof)
    latitudes = _numbers(dataset, "LATITUDE", n_prof)
    longitudes = _numbers(dataset, "LONGITUDE", n_prof)

    out = []

    for i in range(n_prof):
        float_id = _int_or_none(platforms[i])

        if float_id is None:
            continue

        if _int_or_none(position_qc[i]) == _REJECTED:
            continue

        if _int_or_none(time_qc[i]) == _REJECTED:
            continue

        if any(
            _missing(value)
            for value in (juld[i], latitudes[i], longitudes[i])
        ):
            continue

        cycle = _int_or_none(cycles[i])

        if cycle is None:
            continue

        mode = (modes[i] or "R").upper()

        levels = _levels(dataset, i, mode)

        if not levels:
            continue

        latitude = float(latitudes[i])
        longitude = float(longitudes[i])

        out.append(
            ArgoProfile(
                float_id=float_id,
                cycle_number=cycle,
                obs_time=ARGO_EPOCH
                + pd.to_timedelta(float(juld[i]), unit="D"),
                latitude=latitude,
                longitude=longitude,
                region=region_for(latitude, longitude),
                data_mode=mode,
                platform=types[i],
                project=projects[i],
                levels=levels,
            )
        )

    return out


def _levels(dataset, index, mode) -> list[ArgoLevel]:
    """Every level of one profile that measured something at a known depth."""
    values = {}
    flags = {}

    for parameter in PARAMETERS:
        name, qc_name = _variant(dataset, parameter, mode)
        values[parameter] = _column(dataset, name, index)
        flags[parameter] = _qc_column(dataset, qc_name, index)

    out = []

    for level in range(len(values["PRES"])):
        pressure = values["PRES"][level]

        # A measurement with no depth cannot be placed in the water column.
        if _missing(pressure):
            continue

        temperature = values["TEMP"][level]
        salinity = values["PSAL"][level]

        if _missing(temperature) and _missing(salinity):
            continue

        pressure_qc = flags["PRES"][level]
        temperature_qc = flags["TEMP"][level]
        salinity_qc = flags["PSAL"][level]

        out.append(
            ArgoLevel(
                pressure_dbar=float(pressure),
                temperature_c=_clean(temperature),
                salinity_psu=_clean(salinity),
                pressure_qc=pressure_qc,
                temperature_qc=temperature_qc,
                salinity_qc=salinity_qc,
                qc_flag=_overall_qc(
                    [
                        (pressure, pressure_qc),
                        (temperature, temperature_qc),
                        (salinity, salinity_qc),
                    ]
                ),
            )
        )

    return out


def _variant(dataset, parameter, mode) -> tuple[str, str]:
    """The value/QC variable pair this profile's data mode says to read."""
    if mode in ("D", "A") and f"{parameter}_ADJUSTED" in dataset:
        return f"{parameter}_ADJUSTED", f"{parameter}_ADJUSTED_QC"

    return parameter, f"{parameter}_QC"


def _overall_qc(pairs) -> int | None:
    """The worst flag among parameters that actually reported a value.

    A missing salinity should not drag a good temperature row down, so a
    parameter with no value contributes no flag.
    """
    present = [
        flag
        for value, flag in pairs
        if flag is not None and not _missing(value)
    ]

    if not present:
        return None

    return max(
        present,
        key=lambda flag: _QC_BADNESS.get(flag, len(_QC_BADNESS)),
    )


def _column(dataset, name, index) -> np.ndarray:
    """One profile's row of a (N_PROF, N_LEVELS) variable."""
    n_levels = dataset.sizes.get("N_LEVELS", 0)

    if name not in dataset:
        return np.full(n_levels, np.nan)

    return np.atleast_1d(
        np.asarray(dataset[name].values)[index]
    ).astype(float)


def _qc_column(dataset, name, index) -> list[int | None]:
    n_levels = dataset.sizes.get("N_LEVELS", 0)

    if name not in dataset:
        return [None] * n_levels

    row = np.atleast_1d(np.asarray(dataset[name].values)[index])

    return [_int_or_none(_text(item)) for item in row]


def _numbers(dataset, name, n_prof) -> np.ndarray:
    if name not in dataset:
        return np.full(n_prof, np.nan)

    return np.atleast_1d(np.asarray(dataset[name].values))


def _strings(dataset, name, n_prof) -> list[str]:
    """One stripped string per profile, for a char or fixed-width variable."""
    if name not in dataset:
        return [""] * n_prof

    values = np.asarray(dataset[name].values)

    return [_text(values[i]) for i in range(n_prof)]


def _text(value) -> str:
    """Decode a byte, a byte string, or an array of single bytes."""
    if isinstance(value, bytes):
        return value.decode("ascii", "ignore").strip()

    if isinstance(value, str):
        return value.strip()

    parts = [
        item.decode("ascii", "ignore")
        if isinstance(item, bytes)
        else str(item)
        for item in np.asarray(value).ravel()
    ]

    return "".join(parts).strip()


def _int_or_none(value) -> int | None:
    text = _text(value) if not isinstance(value, str) else value.strip()

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _missing(value) -> bool:
    return value is None or bool(pd.isna(value))


def _clean(value) -> float | None:
    return None if _missing(value) else float(value)
