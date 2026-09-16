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

    # Try to make the first column a time axis if it looks like one.
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        except (ValueError, TypeError):
            pass

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