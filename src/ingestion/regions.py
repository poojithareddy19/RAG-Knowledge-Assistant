"""Assign a named region to a profile position.

Shared by every loader so the CSV and NetCDF paths cannot disagree about which
region a position belongs to.

The boundaries are three inequalities, which is coarse for a field that nearly
every query groups by. Replacing them with real basin polygons is on the
roadmap; keeping the rule in one place is what makes that a single edit.
"""

from __future__ import annotations


def region_for(latitude: float, longitude: float) -> str:
    if latitude > 5 and longitude < 78:
        return "Arabian Sea"

    if latitude > 5:
        return "Bay of Bengal"

    return "Southern Indian Ocean"
