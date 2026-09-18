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


def pick_ocean_chart(df):
    """Name the domain plot this result shape supports, or None.

    Dispatch is on column names rather than dtypes, because latitude and
    longitude are just two floats to the generic builder and it will happily
    draw longitude against latitude as a line chart.

    First match wins, and the order is deliberate. A section result carries a
    pressure column too, so it has to be tested before the profile rule, and a
    track carrying temperature and salinity is still a track.
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        return None

    columns = list(df.columns)

    latitude = _find(columns, _LATITUDE)
    longitude = _find(columns, _LONGITUDE)

    if latitude and longitude:
        return "trajectory"

    time = _find(columns, _TIME)
    pressure = _find(columns, _PRESSURE)

    if time and pressure and _value_column(df, (time, pressure)):
        return "section"

    if pressure and _value_column(df, (pressure,)):
        return "profile"

    if _find(columns, _TEMPERATURE) and _find(columns, _SALINITY):
        return "ts_diagram"

    return None


def render_ocean(df, kind, title=None):
    """Return a Plotly figure for the named domain plot."""
    if kind == "trajectory":
        return _trajectory(df, title)

    if kind == "profile":
        return _profile(df, title)

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


def _profile(df, title=None):
    """One vertical cast: the measured value against depth, deepest at the bottom."""
    pressure = _find(df.columns, _PRESSURE)
    value = _value_column(df, (pressure,))

    frame = df.sort_values(pressure)

    figure = go.Figure(
        go.Scatter(
            x=frame[value],
            y=frame[pressure],
            mode="lines+markers",
            name=value,
        )
    )

    figure.update_layout(
        title=title or f"{value} profile",
        xaxis_title=value,
        yaxis_title=pressure,
    )

    figure.update_yaxes(autorange="reversed")

    return figure


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


def _value_column(df, exclude) -> str | None:
    """The first numeric column that is not one of the axes already claimed."""
    for column in df.columns:
        if column in exclude:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            return column

    return None


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
