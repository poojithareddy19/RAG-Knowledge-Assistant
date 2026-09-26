# The database needs two extensions, and no published image carries both.
#
# pgvector holds the summary embeddings; PostGIS holds the basin
# polygons and the profile positions. Dropping either breaks something already
# shipped, so this image adds one to the other.
#
# The base is pgvector and not postgis, which is the opposite of what the task
# asked for, for one reason: glibc. An existing data directory records the
# collation version it was written with, and PostgreSQL warns that text index
# ordering may be wrong when that changes underneath it. pgvector/pgvector:pg16
# is Debian bookworm with glibc 2.36, every published postgis/postgis:16 tag is
# bullseye with 2.31, and swapping to one of those raised exactly that warning
# against the live database. Building the other way round leaves the collation
# untouched.
#
# Pinned to 16 because the data directory is 16, and PostgreSQL refuses to
# start on a data directory written by a different major version.
FROM pgvector/pgvector:pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-16-postgis-3 \
        postgresql-16-postgis-3-scripts \
    && rm -rf /var/lib/apt/lists/*
