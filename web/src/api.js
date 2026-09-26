// One place that knows the shape of the service. Every component below reads
// the object this returns, so a change to the API contract lands here rather
// than in five files.

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export async function ask(question, sessionId, routeOverride = null) {
  const response = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      route_override: routeOverride,
    }),
  })

  if (!response.ok) {
    // The service reports refusals in the body with 200, so a non-200 is a
    // transport or server fault and is worth surfacing verbatim rather than
    // flattening into "something went wrong".
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail.slice(0, 300)}`)
  }

  return response.json()
}

export async function health() {
  const response = await fetch(`${BASE}/health`)
  if (!response.ok) throw new Error(`health ${response.status}`)
  return response.json()
}
