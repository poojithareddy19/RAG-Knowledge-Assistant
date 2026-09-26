import io

import matplotlib

matplotlib.use("Agg")  # No GUI, required on servers

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def to_frame(result):
    return pd.DataFrame(
        result["rows"],
        columns=result["columns"],
    )


# More than this many series and the chart is unreadable anyway, so leave it
# as a table rather than drawing spaghetti.
MAX_SERIES = 12


def spread_categories(df):
    """Turn a long ``(x, category, value)`` result into one column per category.

    A grouped query returns one row per x per group. Plotting that directly
    draws a single line that jumps between groups at every x, which looks like
    a sawtooth and means nothing. Each group has to become its own series.
    """
    if df.shape[1] != 3:
        return df

    x, category, value = df.columns

    if pd.api.types.is_numeric_dtype(df[category]):
        return df

    if not pd.api.types.is_numeric_dtype(df[value]):
        return df

    if df[category].nunique() > MAX_SERIES:
        return df

    wide = df.pivot_table(
        index=x,
        columns=category,
        values=value,
        aggfunc="mean",
    )

    wide.columns.name = None

    return wide.reset_index()


# A step this many times the usual one is a hole in the data, not a slow year.
GAP_FACTOR = 1.5


def usual_step(x):
    """The spacing of a time or numeric axis, or None if it has none.

    A calendar bucket is recognised rather than estimated: dates that are all
    1 January are yearly, dates that are all the 1st are monthly, and whole
    numbers between 1800 and 2200 are years. The median is only the fallback,
    because a series with many holes has a misleading one: the Arabian Sea's
    profiles per year skip so often that the median step came out at a year
    and a half, and a missing 2024 was drawn straight across.
    """
    if len(x) < 4:
        return None

    if pd.api.types.is_datetime64_any_dtype(x):
        if (x.dt.month == 1).all() and (x.dt.day == 1).all():
            return pd.Timedelta(days=365.25)

        if (x.dt.day == 1).all():
            return pd.Timedelta(days=30.44)

    elif pd.api.types.is_numeric_dtype(x):
        if (x % 1 == 0).all() and x.between(1800, 2200).all():
            return 1

    else:
        return None

    usual = x.sort_values().diff().iloc[1:].median()

    # usual - usual is zero of the right kind, a Timedelta or a number.
    if pd.isna(usual) or usual <= usual - usual:
        return None

    return usual


def find_gaps(x):
    """The (start, end) pairs where a regular series skips ahead.

    A line drawn straight across a gap claims values for years nobody measured:
    profiles per year ran from 2009 to 2023 as a smooth rise, over thirteen
    years with no profiles at all. The usual step is the median one, so a
    yearly or monthly series defines its own spacing, and anything over
    ``GAP_FACTOR`` times it is a gap. Fewer than four points cannot establish
    a step, and an axis that is neither time nor numbers has no spacing.
    """
    usual = usual_step(x)

    if usual is None:
        return []

    ordered = x.sort_values().reset_index(drop=True)
    steps = ordered.diff().iloc[1:]

    return [
        (ordered.iloc[i - 1], ordered.iloc[i])
        for i, step in steps.items()
        if step > usual * GAP_FACTOR
    ]


def break_at_gaps(df, x, gaps):
    """Insert an empty row inside each gap, which matplotlib draws as a break."""
    if not gaps:
        return df

    holes = pd.DataFrame(
        {x: [start + (end - start) / 2 for start, end in gaps]},
    )

    return (
        pd.concat([df, holes], ignore_index=True)
        .sort_values(x)
        .reset_index(drop=True)
    )


def _as_time_axis(column):
    """A time column the query returned as objects, made a real datetime axis.

    ``date_trunc`` on a timestamptz arrives as timezone-aware Python objects,
    which pandas keeps as an object column that neither the gap check nor
    matplotlib's date axis recognises. The time zone is dropped because the
    axis is labelled in years and months, where it changes nothing.
    """
    if not pd.api.types.is_object_dtype(column):
        return column

    try:
        converted = pd.to_datetime(column, utc=True)
    except (ValueError, TypeError):
        return column

    return converted.dt.tz_localize(None)


def pick_chart(df):
    """Decide a chart type from the shape of the data."""
    if df.shape[1] < 2:
        return "table"

    first = df.iloc[:, 0]

    if pd.api.types.is_datetime64_any_dtype(first):
        return "line"

    if pd.api.types.is_numeric_dtype(first) and df.shape[0] > 12:
        return "line"

    return "bar"


def render(result, title=None):
    """Return (png_bytes, chart_kind).

    None means the data is not chartable.
    """
    df = to_frame(result)

    if df.empty:
        return None, "empty"

    # Try to make the first column a time axis if it looks like one. The whole
    # column is replaced rather than assigned into with iloc, which keeps the
    # old object dtype and leaves the axis unrecognised as time.
    first = df.columns[0]

    if not pd.api.types.is_numeric_dtype(df[first]):
        df[first] = _as_time_axis(df[first])

    df = spread_categories(df)

    kind = pick_chart(df)

    if kind == "table":
        return None, "table"

    x = df.columns[0]

    ycols = [
        column
        for column in df.columns[1:]
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if not ycols:
        return None, "table"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if kind == "line":
        # The step is read from the real points, before the break rows go in:
        # those sit mid-gap, off the calendar grid, and would push the step
        # onto the median fallback.
        step = usual_step(df[x])
        gaps = find_gaps(df[x])
        df = break_at_gaps(df, x, gaps)

        # Shaded as well as broken: a break alone reads as a rendering fault.
        # Each band stops half a step short of the points on either side, so a
        # lone year inside a long gap sits on white rather than looking as if
        # it were part of the missing stretch, and two neighbouring gaps show
        # as two bands. A series too short to have a step has no gaps, so the
        # inset is only needed inside the loop; computing it outside divided
        # None by two and turned a three-point chart into a 500.
        for start, end in gaps:
            inset = step / 2
            ax.axvspan(start + inset, end - inset, color="0.92", zorder=0)
            ax.text(
                start + (end - start) / 2,
                0.97,
                "no data",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="0.45",
            )

    for column in ycols:
        if kind == "line":
            ax.plot(
                df[x],
                df[column],
                marker="o",
                markersize=3,
                label=column,
            )
        else:
            ax.bar(
                df[x].astype(str),
                df[column],
                label=column,
            )

    ax.set_xlabel(x)
    ax.set_title(title or "")

    if len(ycols) > 1:
        ax.legend()

    if kind == "bar":
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=120,
    )
    plt.close(fig)

    return buf.getvalue(), kind