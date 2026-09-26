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

## Where the images come from

CI builds both on every push and publishes them on `main`:

```
ghcr.io/poojithareddy19/floatchat-api:latest     also :<commit sha>
ghcr.io/poojithareddy19/floatchat-web:latest     also :<commit sha>
```

The `images` job runs only after lint and the test suite pass, so a published
image is one whose commit was green. The sha tag is the one to deploy: `latest`
tells you nothing about what you are running six weeks later, and the sha does.

A pull request from a fork still builds both images to prove they build, and
pushes nothing, because it has no token that can write packages and does not
need one for that.

This is deliberately where delivery stops. The images a server would pull
exist, are built from an exact commit, and are tagged with it. Nothing pulls
them onto a host, and no cloud account is involved.

## Bring it up

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Naming both files matters. `docker-compose.override.yml` is picked up
automatically and is local-only: it publishes the database on 5433 because
5432 is taken on the development machine, which a server has no reason to copy.

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
| `POSTGRES_PASSWORD` | `gda_local_pw` | A literal in `docker-compose.yml`, not read from the environment: edit the file. The read-only role's password is set in `db/003_roles.sql` |
| `DATABASE_URL` | the same | Both roles' passwords are the committed defaults |
| `WEB_PORT` | `80` | Put a TLS terminator in front, or change the port |
| `GENERATION_MODEL` | `llama3.1:latest` | |

The read-only role is not a formality. Generated SQL executes as `gda_ro`,
which is what makes the validator a second line rather than the only one.

## Health

`GET /health` reports the number of summaries in the semantic index, the
measurement count and whether the model is reachable. Both images carry a `HEALTHCHECK`, and `web` waits for the
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
- **Not on a cloud host.** By choice. Everything up to that point is proven,
  and the section below says how.

## What was verified, and what it found

Both images were built and the production arrangement brought up on the
machine that wrote it, against the real archive:

| Check | Result |
| --- | --- |
| `docker build` of the API image | 3.43 GB, torch from the CPU index |
| `docker build` of the front end | 73.8 MB, bundle present, `nginx -t` passes standalone |
| `GET /api/health` through nginx | 175,364 measurements, model reachable (the health body named the manual chunks at the time; it names the summaries now) |
| `POST /api/ask` through nginx | HTTP 200 in 8 s, the scope gate's refusal, request seen by the API from the web container's address |
| `GET /` | the front end, with its fingerprinted bundle |
| `docker compose ps` | all three services healthy, only port 80 published |

`docker compose config` had passed all of this before any of it was run, and
running it found three defects that config validation cannot see:

1. **The API image pulled the CUDA build of torch.** `sentence-transformers`
   brings in torch, and the default wheel on Linux carries the NVIDIA runtime,
   about two gigabytes, into a container with no GPU. The first build died in
   that install with pip exit code 2, out of memory. CI already used the CPU
   wheel index; the Dockerfile did not. An image that only builds on the CI
   runner is not a deployable image.

2. **nginx refused to start without the API.** `proxy_pass http://api:8000/`
   resolves the name once, at startup, and fails with `host not found in
   upstream "api"` if it is not there. Compose's `depends_on` masked it, but it
   meant the front end could not be tested alone and a restarted API took the
   front end down with it. The upstream is now a variable resolved per request
   through Docker's DNS, so the front end always serves and `/api` answers 502
   for the seconds the service is away.

3. **The front end's healthcheck reported a working server as down.** It asked
   `http://localhost/`, which inside the container resolves to `::1` first,
   and nginx listens on IPv4 only. Every real request succeeded while the
   check said unhealthy. It asks `127.0.0.1` now. A healthcheck that disagrees
   with the service it checks is worse than none, because it teaches you to
   ignore red.

None of the three would have been found by reading the files.
