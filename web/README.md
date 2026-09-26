# FloatChat web

The front end for FloatChat, a React client of the FastAPI service. An answer
you cannot check is not an answer, so every turn shows the route that was
taken, the SQL that produced the table, the chart drawn from it, and the float
or region behind a summary claim.

## Running

The API has to be up first, because this talks to it rather than to the
database:

```bash
uvicorn src.api.main:app --reload     # from the repository root
```

Then:

```bash
cd web
npm install
npm run dev
```

http://localhost:5173. Vite proxies `/api` to `http://localhost:8000`, so the
browser stays on one origin and there is no CORS configuration to get wrong.
Point it elsewhere with `VITE_API_BASE` if the service is not local.

## Why only through the API

It talks to `POST /ask` and to nothing else, as a client on another server
would. That is the point of having it: a UI written against `AskResponse` finds
the places where that response is awkward to consume, which a page reaching
into the pipeline directly never can. It is the only front end; there used to
be a Streamlit dashboard beside it, removed so that behaviour is reachable one
way. Evaluation runs from the command line (`src/evaluation/`) and monitoring
reads `logs/interactions.jsonl` and `logs/traces.jsonl`.

## Contract

One endpoint does the work. `POST /ask` takes

```json
{ "question": "...", "session_id": "...", "route_override": null }
```

and returns `AskResponse`: the answer, the route and who decided it, the
confidence, and then whichever evidence applies. A summary answer carries
`citations`, each naming the float or region it rests on; a data answer
carries `generated_sql`, `columns` and `rows`, and a chart when the rows have
a shape worth drawing: `chart_spec`, a Plotly figure for trajectories, profiles,
sections and T-S diagrams, or `chart_png_base64` from the generic renderer when
a chart was asked for and the rows are an ordinary aggregate. A
refusal sets `refused` with a `reason` and is rendered as a result rather than
an error, because the system declining is the behaviour this project wants.

Passing the same `session_id` is what makes a follow up like "and in 2022?"
resolvable. It is generated per page load rather than persisted: a stale id
would rewrite a new question against a conversation the user has forgotten.

## Layout

```
src/
  api.js                  the only file that knows the service's shape
  App.jsx                 question box, turn list, session id
  components/
    Turn.jsx              one question and everything shown for it
    RouteBadge.jsx        which half of the system answered
    SqlBlock.jsx          the query behind the table, one click away
    ResultTable.jsx       first 100 rows, with the true count
    Chart.jsx             the Plotly figure or the PNG; Plotly loads on first use
    Citations.jsx         the float or region behind a claim, with similarity
public/
  topojson/               world basemaps, so trajectory maps draw offline
```
