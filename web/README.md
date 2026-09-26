# FloatChat web

A React front end over the FastAPI service, for the same reason the Streamlit
page exists: an answer you cannot check is not an answer. Every turn shows the
route that was taken, the SQL that produced the table, and the
float or region behind a summary claim.

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

## What it is not

Not a replacement for the Streamlit app. That one has the evaluation and
monitoring pages and the interactive ocean charts, and it is where the
project is actually used. This exists because the problem statement asks for a
React front end on the production path, and because the API contract is worth
having a second client for: a UI written against `AskResponse` finds the places
where that response is awkward, and the Streamlit page cannot, since it reaches
past the API into the pipeline directly.

## Contract

One endpoint does the work. `POST /ask` takes

```json
{ "question": "...", "session_id": "...", "route_override": null }
```

and returns `AskResponse`: the answer, the route and who decided it, the
confidence, and then whichever evidence applies. A summary answer carries
`citations`, each naming the float or region it rests on; a data answer
carries `generated_sql`, `columns` and `rows`. A
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
    Citations.jsx         the float or region behind a claim, with similarity
```
