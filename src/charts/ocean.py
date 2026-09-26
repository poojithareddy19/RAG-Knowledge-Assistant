"""Domain plots for ocean data: trajectories, profiles, sections, T-S diagrams.

These are the four views the field actually reads. The generic builder in
[`builder.py`](builder.py) decides what to draw from dtypes, which is the right
default for an arbitrary aggregate and exactly wrong here: latitude and
longitude are two floats to a dtype check, so a track gets drawn as a line of
longitude against latitude, and a depth profile gets drawn upside down.

So dispatch is on column names, and it runs before the generic builder rather
than instead of it. Anything this module does not recognise falls through
untouched.

The one convention worth stating: depth increases downward. A profile or a
section with a normal y axis is read as wrong by anyone in the field, so the
axis is reversed in both.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go

# Substring matched, case insensitively, so a query aliasing a column as
# avg_temperature or p.latitude still lands on the right plot.
_LATITUDE = ("latitude",)
_LONGITUDE = ("longitude",)
_TIME = ("time", "date", "juld")
_PRESSURE = ("pressure", "depth", "dbar")
_TEMPERATURE = ("temperature", "temp")
_SALINITY = ("salinity", "psal")
_FLOAT = ("float_id", "float", "platform")

# Numeric, and never the thing being measured. The value column used to be the
# first numeric column not already claimed as an axis, so a result carrying
# profile_id or cycle_number drew the identifier against depth as though it
# were salinity. Positions and QC flags are numeric too and are not values.
#
# Matched on whole parts of the name split at underscores, not as substrings:
# "id" as a substring would also exclude mid_depth, and "lat" relative_error.
_NOT_A_VALUE = frozenset({
    "id", "cycle", "number", "qc", "flag", "count", "wmo", "rank",
    "distance", "km", "lat", "lon", "latitude", "longitude",
})

# Preferred, in order, when a result carries several measured columns.
_MEASURED = (
    "salinity", "psal", "temperature", "temp", "sst", "oxygen", "doxy",
    "chlorophyll", "chla", "nitrate", "ph", "backscatter", "speed", "velocity",
)

# Keys that identify one cast, best first. One trace per cast is what makes a
# profile plot a comparison rather than a zigzag.
_CAST_KEYS = (("profile_id",), ("float_id", "cycle_number"), ("float_id",), ("obs_time",))

# More casts than this on one axis is a smear rather than a comparison, and the
# legend stops being readable well before it.
MAX_PROFILES = 25

_WANTS_PROFILES = ("profile", "compare", "comparison", "versus", " vs")


def pick_ocean_chart(df, question=None):
    """Name the domain plot this result shape supports, or None.

    Dispatch is on column names rather than dtypes, because latitude and
    longitude are just two floats to the generic builder and it will happily
    draw longitude against latitude as a line chart.

    First match wins, and the order is deliberate. A result carrying pressure
    and a measured value is vertical structure, and it is tested before the map
    rule: it used to be the other way round, so a profile query that also
    returned each cast's position, as nearly all of them do, was drawn as a map
    of where the float surfaced and the profile itself was never shown. A track
    carries positions and no pressure, and is still a track.

    ``question`` settles the one ambiguous case. Casts at several times can be
    drawn as profiles overlaid or as a depth-time section, and both are right;
    a question that asks for profiles, or to compare, gets the overlay.
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        return None

    columns = list(df.columns)

    time = _find(columns, _TIME)
    pressure = _find(columns, _PRESSURE)

    if pressure and _value_column(df, (pressure, time)):
        wants_profiles = any(
            word in f" {question or ''}".lower() for word in _WANTS_PROFILES
        )

        several_times = bool(time) and _distinct(df, time) > 1

        if several_times and not wants_profiles:
            return "section"

        return "profile"

    latitude = _find(columns, _LATITUDE)
    longitude = _find(columns, _LONGITUDE)

    if latitude and longitude:
        return "trajectory"

    if _find(columns, _TEMPERATURE) and _find(columns, _SALINITY):
        return "ts_diagram"

    return None


def _distinct(df, column) -> int:
    try:
        return int(df[column].nunique(dropna=True))
    except Exception:
        return 0


def render_ocean(df, kind, title=None, truncated=False):
    """Return a Plotly figure for the named domain plot.

    ``truncated`` says the rows filled the query's LIMIT, so the last cast in a
    profile result may be cut off partway down.
    """
    if kind == "trajectory":
        return _trajectory(df, title)

    if kind == "profile":
        return _profile(df, title, truncated=truncated)

    if kind == "section":
        return _section(df, title)

    if kind == "ts_diagram":
        return _ts_diagram(df, title)

    raise ValueError(f"unknown ocean chart kind: {kind}")


def _trajectory(df, title=None):
    """Where the floats went, one trace per float, points ordered by time.

    Without a float id column the points are drawn unconnected. Joining the
    positions of two different floats invents a path that nothing travelled,
    and the result looks like data rather than like an artefact of plotting.
    """
    latitude = _find(df.columns, _LATITUDE)
    longitude = _find(df.columns, _LONGITUDE)
    time = _find(df.columns, _TIME)
    floats = _find(df.columns, _FLOAT)

    frame = df.copy()

    if time:
        frame = _sorted_by_time(frame, time)

    figure = go.Figure()

    groups = (
        frame.groupby(floats, sort=True)
        if floats
        else [(None, frame)]
    )

    for name, group in groups:
        figure.add_trace(
            go.Scattergeo(
                lat=group[latitude],
                lon=group[longitude],
                mode="lines+markers" if floats else "markers",
                name=str(name) if floats else "positions",
                text=(
                    group[time].astype(str)
                    if time
                    else None
                ),
                marker=_time_marker(group, time),
            )
        )

    figure.update_layout(
        title=title or "Float trajectory",
        geo={
            "showland": True,
            "landcolor": "rgb(243, 243, 243)",
            "showcountries": True,
            "fitbounds": "locations",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return figure


def _profile(df, title=None, truncated=False):
    """Vertical casts: the measured value against depth, one line per cast.

    This drew one line through every row sorted by pressure. For one cast that
    is right. For several, which is what "salinity profiles near the equator"
    returns, it joined the 5 dbar reading of one float to the 5 dbar reading of
    the next and zigzagged between casts all the way down, producing a shape no
    water column has. Each cast now gets its own trace, so several profiles are
    a comparison, which the problem statement asks for by name.
    """
    pressure = _find(df.columns, _PRESSURE)
    time = _find(df.columns, _TIME)
    value = _value_column(df, (pressure, time))
    key = _cast_key(df)

    figure = go.Figure()

    if key is None:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(key, sort=True))

    # A result that filled its LIMIT was cut somewhere, and ordered by cast
    # then depth the cut falls inside the last cast. Drawn, it would show a
    # float that stopped halfway down. Dropped, and the title says so, because
    # an incomplete cast presented as a complete one is a fabricated profile.
    cut = truncated and key is not None and len(groups) > 1

    if cut:
        groups = groups[:-1]

    # One cast that filled the limit on its own cannot be dropped, because it
    # is all there is. It is drawn, and the title says it may stop short.
    partial = truncated and not cut

    shown = groups[:MAX_PROFILES]

    for name, group in shown:
        frame = group.sort_values(pressure)

        figure.add_trace(
            go.Scatter(
                x=frame[value],
                y=frame[pressure],
                mode="lines+markers",
                name=_label(key, name) if key else value,
                marker={"size": 4},
            )
        )

    heading = title or f"{value} profile{'s' if len(groups) > 1 else ''}"

    if len(groups) > MAX_PROFILES:
        heading += f" (first {MAX_PROFILES} of {len(groups)} casts)"

    if cut:
        heading += " (row limit reached: the last, incomplete cast is not drawn)"

    if partial:
        heading += " (row limit reached: this cast may stop short of its full depth)"

    figure.update_layout(
        title=heading,
        xaxis_title=value,
        yaxis_title=pressure,
        showlegend=len(shown) > 1,
    )

    figure.update_yaxes(autorange="reversed")

    return figure


def _cast_key(df):
    """The columns that identify one cast in this result, or None."""
    lowered = {str(column).lower(): column for column in df.columns}

    for candidate in _CAST_KEYS:
        if all(part in lowered for part in candidate):
            columns = [lowered[part] for part in candidate]

            # A key with one value is not a grouping, it is a constant.
            if _distinct(df, columns[0]) > 1 or len(columns) > 1:
                return columns if len(columns) > 1 else columns[0]

    return None


def _label(key, name) -> str:
    if isinstance(key, list):
        parts = name if isinstance(name, tuple) else (name,)
        return " · ".join(f"{k} {v}" for k, v in zip(key, parts, strict=False))

    return f"{key} {name}"


def _section(df, title=None):
    """Depth against time with the measurement as colour: a vertical slice."""
    time = _find(df.columns, _TIME)
    pressure = _find(df.columns, _PRESSURE)
    value = _value_column(df, (time, pressure))

    frame = _sorted_by_time(df, time)

    figure = go.Figure(
        go.Scatter(
            x=frame[time],
            y=frame[pressure],
            mode="markers",
            marker={
                "color": frame[value],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": value},
                "size": 6,
            },
            text=frame[value].astype(str),
            name=value,
        )
    )

    figure.update_layout(
        title=title or f"{value} section",
        xaxis_title=time,
        yaxis_title=pressure,
    )

    figure.update_yaxes(autorange="reversed")

    return figure


def _ts_diagram(df, title=None):
    """Temperature against salinity, as points.

    Points and not lines: a T-S diagram identifies water masses by where the
    cloud sits, and connecting the observations in row order draws a path
    through the water column that means nothing.
    """
    temperature = _find(df.columns, _TEMPERATURE)
    salinity = _find(df.columns, _SALINITY)
    pressure = _find(df.columns, _PRESSURE)

    marker = {"size": 5}

    if pressure:
        marker = {
            "size": 5,
            "color": df[pressure],
            "colorscale": "Viridis_r",
            "showscale": True,
            "colorbar": {"title": pressure},
        }

    figure = go.Figure(
        go.Scatter(
            x=df[salinity],
            y=df[temperature],
            mode="markers",
            marker=marker,
            name="T-S",
        )
    )

    figure.update_layout(
        title=title or "Temperature against salinity",
        xaxis_title=salinity,
        yaxis_title=temperature,
    )

    return figure


def _find(columns, needles) -> str | None:
    """First column whose name contains one of ``needles``, case insensitively."""
    for needle in needles:
        for column in columns:
            if needle in str(column).lower():
                return column

    return None


def _is_label(column) -> bool:
    """An identifier, position, flag or tally rather than a measurement."""
    parts = re.split(r"[^a-z0-9]+", str(column).lower())

    return any(part in _NOT_A_VALUE for part in parts)


def _value_column(df, exclude) -> str | None:
    """The measured quantity to plot: never an axis, an identifier or a flag.

    A named measurement is preferred over any other numeric column, so a
    result carrying both salinity and a count plots the salinity.
    """
    candidates = [
        column
        for column in df.columns
        if column not in exclude
        and pd.api.types.is_numeric_dtype(df[column])
        and not _is_label(column)
    ]

    for wanted in _MEASURED:
        for column in candidates:
            if wanted in str(column).lower():
                return column

    return candidates[0] if candidates else None


def _sorted_by_time(df, time):
    """Rows in time order, leaving them alone if the column will not parse.

    A track drawn in result order rather than time order joins the points in
    whatever sequence the database happened to return, which is a different
    path from the one the float took.
    """
    frame = df.copy()

    try:
        frame[time] = pd.to_datetime(frame[time])
    except (ValueError, TypeError):
        return df

    return frame.sort_values(time)


def _time_marker(group, time):
    """Marker spec colouring the points by time, or a plain one without it."""
    if not time:
        return {"size": 6}

    stamps = pd.to_datetime(group[time], errors="coerce")

    if stamps.isna().all():
        return {"size": 6}

    return {
        "size": 6,
        "color": stamps.astype("int64"),
        "colorscale": "Viridis",
        "showscale": False,
    }
