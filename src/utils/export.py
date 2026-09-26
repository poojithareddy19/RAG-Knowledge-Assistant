"""Serialise a result set to CSV or to self describing NetCDF.

CSV is the format everyone can open and the format that tells you nothing: a
column called `mean_temp` holding 25.37 does not say whether that is Celsius,
Kelvin or Fahrenheit, and the query that produced it is gone the moment the
file leaves the page.

NetCDF is what this domain already reads, and it carries attributes, so the
download can say what its variables mean and which query made them. A NetCDF
without attributes is a CSV that is harder to open, so the attributes are the
point of this module rather than a nicety.

Units are not inferred from the data or parsed out of English. The catalog
states them in prose, which is right for a model reading a prompt and wrong as
a source for a machine readable attribute, so the mapping below is explicit and
a column that is not in it simply has no units attribute. An absent unit is a
gap; a guessed one is a wrong number waiting to be plotted.
"""

from __future__ import annotations

import datetime as dt
import re
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import xarray as xr

from src.sqlgen.schema_context import load_catalog

# CF style unit strings for the columns this schema defines, transcribed from
# db/schema_catalog.md. Keyed on the column name the database uses; a query
# aliasing a column to something else loses its unit, which is the honest
# outcome, because avg(temperature_c) AS mean_temp is still Celsius but
# avg(temperature_c) / 2 AS mean_temp is not and nothing here can tell.
UNITS = {
    "pressure_dbar": "dbar",
    "temperature_c": "degree_Celsius",
    "salinity_psu": "psu",
    "oxygen_umol_kg": "umol kg-1",
    "chlorophyll_mg_m3": "mg m-3",
    "nitrate_umol_kg": "umol kg-1",
    "ph_total": "1",
    "backscatter_700": "m-1",
    "latitude": "degrees_north",
    "longitude": "degrees_east",
}

_BULLET = re.compile(r"^\s*-\s+(`[^`]+`(?:\s*,\s*`[^`]+`)*)\s*(\([^)]*\))?\s*(?:-\s*(.*))?$")
_NAME = re.compile(r"`([^`]+)`")


def to_csv_bytes(result) -> bytes:
    """The result set as UTF-8 CSV, header included."""
    return to_frame(result).to_csv(index=False).encode("utf-8")


def to_parquet_bytes(result, title="ARGO query result") -> bytes:
    """The result set as Parquet, with the question and units in its metadata.

    The problem statement names Parquet beside SQL as a target format, and it
    is the one of the three an analyst's pandas or Spark session reads without
    parsing: columnar, typed, compressed. Types are kept rather than turned
    into text, so a timestamp arrives as a timestamp and a count as an integer.

    Parquet has no per-column attributes the way NetCDF does, so the units and
    descriptions from the catalog go into the file's key-value metadata, along
    with the question and the SQL that produced the rows. A file separated from
    this page still says what it is and how to reproduce it.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = _clean(to_frame(result))
    table = pa.Table.from_pandas(frame, preserve_index=False)

    described = column_descriptions()

    metadata = {
        **(table.schema.metadata or {}),
        b"floatchat.title": str(title).encode("utf-8"),
        b"floatchat.columns": json.dumps(
            {c: described.get(c, "") for c in frame.columns}
        ).encode("utf-8"),
    }

    sql = result.get("generated_sql")

    if sql:
        metadata[b"floatchat.sql"] = str(sql).encode("utf-8")

    buffer = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata(metadata), buffer)

    return buffer.getvalue().to_pybytes()


def to_netcdf_bytes(result, title="ARGO query result") -> bytes:
    """Serialise a result set to NetCDF with units attached.

    Units and long names come from the column catalog, so a downloaded file
    says what its variables mean. A NetCDF without attributes is a CSV that
    is harder to open.
    """
    dataset = to_dataset(result, title=title)

    return _write(dataset)


def to_frame(result) -> pd.DataFrame:
    """The rows and columns of a result as a DataFrame."""
    return pd.DataFrame(
        result.get("rows") or [],
        columns=result.get("columns") or [],
    )


def to_dataset(result, title="ARGO query result") -> xr.Dataset:
    """Build the xarray dataset, attributes and all.

    Kept separate from the bytes so a test can read the structure back without
    a round trip through a file, and so a caller wanting xarray directly does
    not have to serialise and reopen.
    """
    frame = _clean(to_frame(result))

    dataset = _shape(frame)

    catalog = column_descriptions()

    for name in dataset.variables:
        column = str(name)

        dataset[name].attrs["long_name"] = catalog.get(column, column)

        if column in UNITS:
            dataset[name].attrs["units"] = UNITS[column]

    dataset.attrs["title"] = title
    dataset.attrs["sql"] = str(result.get("generated_sql") or "")
    dataset.attrs["created"] = (
        dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    dataset.attrs["source"] = "Argo GDAC profiles, loaded by this project"

    return dataset


def column_descriptions() -> dict[str, str]:
    """``{column: description}`` scraped from the markdown column catalog.

    The catalog is the one place the columns are already described, and
    duplicating those sentences here would mean two of them to keep in step.
    A line naming several columns at once gives all of them the same
    description, which is what that line means.
    """
    try:
        text = load_catalog()
    except Exception:
        return {}

    out = {}

    for line in _bullets(text):
        match = _BULLET.match(line)

        if not match:
            continue

        names, _, description = match.groups()

        if not description:
            continue

        for name in _NAME.findall(names):
            out.setdefault(name.strip(), description.strip())

    return out


def _bullets(text: str) -> list[str]:
    """Markdown bullets reflowed onto one line each.

    The catalog wraps at the margin, and several of the wrapped bullets are the
    ones naming a group of columns together, so a line by line parse silently
    loses exactly the entries a reader most needs.
    """
    out: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("- "):
            out.append(stripped)
        elif out and stripped and line.startswith((" ", "\t")):
            out[-1] = f"{out[-1]} {stripped}"
        else:
            # A blank line or a heading ends the bullet it followed.
            out.append("")

    return [line for line in out if line]


def _shape(frame: pd.DataFrame) -> xr.Dataset:
    """Index by profile and pressure when both are present, else by row.

    A result carrying a profile identifier and a pressure is a set of casts,
    and indexing it that way is what lets a reader select one profile or slice
    a depth range. Anything else is a table, and a table's only honest
    dimension is the row it came back on.
    """
    profile = _match(frame.columns, ("profile_id",))
    pressure = _match(frame.columns, ("pressure",))

    if profile and pressure:
        indexed = frame.set_index([profile, pressure])

        # from_dataframe cannot build a grid from repeated coordinates, and a
        # result with two rows for one profile at one depth is a table.
        if indexed.index.is_unique:
            return xr.Dataset.from_dataframe(indexed)

    return xr.Dataset(
        {
            str(column): ("obs", frame[column].to_numpy())
            for column in frame.columns
        },
        coords={"obs": range(len(frame))},
    )


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the types psycopg returns into ones netCDF can hold."""
    out = frame.copy()

    for column in out.columns:
        series = out[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            out[column] = _naive_utc(series)
            continue

        if series.map(lambda value: isinstance(value, Decimal)).any():
            out[column] = pd.to_numeric(series, errors="coerce")
            continue

        if pd.api.types.is_object_dtype(series):
            converted = pd.to_datetime(series, errors="coerce", utc=True)

            if converted.notna().all() and len(series):
                out[column] = _naive_utc(converted)
                continue

            out[column] = series.astype(str)

    return out


def _naive_utc(series: pd.Series) -> pd.Series:
    """Timestamps in UTC without a timezone, which is all netCDF can store."""
    if getattr(series.dtype, "tz", None) is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)

    return series


def _match(columns, needles) -> str | None:
    for needle in needles:
        for column in columns:
            if needle in str(column).lower():
                return column

    return None


def _write(dataset: xr.Dataset) -> bytes:
    """Bytes for the dataset, through a file when the backend insists.

    scipy's writer returns bytes from an in-memory call, netCDF4's does not and
    raises instead, so the temporary file is the fallback rather than the
    default: it costs a write and a read that the buffer path does not.
    """
    try:
        blob = dataset.to_netcdf()

        if isinstance(blob, bytes | bytearray | memoryview):
            return bytes(blob)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.nc"

        dataset.to_netcdf(path)

        return path.read_bytes()


def open_netcdf_bytes(blob: bytes) -> xr.Dataset:
    """Reopen exported bytes, which is what a caller checking a download does.

    Through a file rather than a buffer, because the netCDF4 backend reads a
    path and the in-memory readers are a separate optional dependency. The
    dataset is loaded eagerly so it survives the file being deleted.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.nc"
        path.write_bytes(blob)

        with xr.open_dataset(path) as dataset:
            return dataset.load()
