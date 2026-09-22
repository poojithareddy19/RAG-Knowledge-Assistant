# Deploying FloatChat

Three containers and a model server. The database and the API are internal;
nginx is the only thing listening.

```
        :80
         │
    ┌────▼────┐      ┌─────────┐      ┌──────────┐
    │   web   │─────▶│   api   │─────▶│    db    │
    │  nginx  │ /api │ uvicorn │      │ postgis  │
    └─────────┘      └────┬────┘      │ pgvector │
                          │           └──────────┘
                          ▼
                   Ollama, on the host
```

## Why Ollama is not a container here

It needs the host's memory and, where there is one, the host's GPU. Putting it
in the compose file would work on a laptop and fall over on anything with an
accelerator, so it runs on the host and the API reaches it through
`host.docker.internal`.

If your model server is elsewhere, set `OLLAMA_URL` and uncomment the line the
`api` service points at. **Do not set `OLLAMA_BASE_URL` in `.env` for a
container deployment.** Compose interpolates from `.env`, where that variable
is `http://localhost:11434` because that is correct for the loaders and the
evaluation harness, which run on the host. Inside a container `localhost` is
the container, so the API would be pointed at itself. This is written down
because it was got wrong once while writing this file, and it fails silently:
the container starts, health is green, and every question times out.

## Bring it up

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Naming both files matters. `docker-compose.override.yml` is picked up
automatically and is local-only: it carries bind mounts of the working tree and
publishes the database on 5433. On a server that would serve whatever happens
to be in the checkout.

Add the Streamlit dashboard, which carries the evaluation and monitoring pages,
with `--profile dashboard`. It is for whoever runs this rather than whoever
asks it questions.

## Load the data

The image does not carry the archive. On a new volume the schema is created by
the init scripts, and everything else is loaded from the host:

```bash
python scripts/fetch_argo_index.py
python scripts/load_argo_netcdf.py
python scripts/load_regions.py
python scripts/load_drifters.py --year 2023
python scripts/build_semantic_index.py
```

Point `DATABASE__URL` at the deployed database first. Loading is the one
operation that writes, so it uses the owning role rather than the read-only one
generated SQL runs as.

## Settings worth changing before this faces anyone

| Variable | Default | Why |
| --- | --- | --- |
| `POSTGRES_PASSWORD` | `gda_local_pw` | In `docker-compose.yml`, in plain text, in the repository |
| `DATABASE_URL` | the same | Both roles' passwords are the committed defaults |
| `WEB_PORT` | `80` | Put a TLS terminator in front, or change the port |
| `GENERATION_MODEL` | `llama3.1:latest` | |

The read-only role is not a formality. Generated SQL executes as `gda_ro`,
which is what makes the validator a second line rather than the only one.

## Health

`GET /health` reports indexed documents, measurement count and whether the
model is reachable. Both images carry a `HEALTHCHECK`, and `web` waits for the
API to be healthy rather than merely started.

An orchestrator wanting a liveness probe should use `/health` on the API and
`/` on nginx. Give the API a generous start period: it loads an embedding model
before it serves anything.

## What this does not do

- **No TLS.** nginx listens on 80. Terminate TLS in front of it.
- **No autoscaling, no managed database.** One compose file on one host. For
  ECS or similar, the images are the unit of work and the arrangement above is
  the shape; the compose file is not a task definition.
- **No secret management.** Passwords come from the environment, and the
  defaults are committed.
- **Not load tested.** Generation is single-threaded through one Ollama
  instance. Concurrent questions queue, and on modest hardware each takes tens
  of seconds, so this arrangement serves a small number of users.
- **Not deployed anywhere.** These files describe a deployment that has been
  validated with `docker compose config` and has not been run on a server. The
  images have not been built on this machine, which is short of disk. Treat the
  first deploy as the first test.
