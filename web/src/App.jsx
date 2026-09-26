import { useEffect, useRef, useState } from 'react'
import { ask, health } from './api.js'
import Turn from './components/Turn.jsx'

// The same session id travels with every question, which is what makes
// "and in 2022?" resolvable. Generated once per page load rather than stored:
// a stale id would rewrite a new question against a conversation the user has
// forgotten having.
const SESSION = crypto.randomUUID()

const EXAMPLES = [
  'What is the average surface temperature in the Arabian Sea?',
  'Tell me about float 1901393',
  'How fast is the surface current on average in the Arabian Sea?',
  'What is the current speed at 1000 decibars?',
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, busy])

  async function submit(event, preset) {
    event?.preventDefault()

    const text = (preset ?? question).trim()
    if (!text || busy) return

    setQuestion('')
    setBusy(true)

    try {
      const result = await ask(text, SESSION)
      setTurns((t) => [...t, { question: text, result }])
    } catch (err) {
      setTurns((t) => [...t, { question: text, error: String(err.message ?? err) }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>FloatChat</h1>
        <p className="tagline">
          Ask the Argo float archive a question and get an answer you can check.
        </p>
        {status && (
          <p className="muted status">
            {status.summaries_indexed ?? 0} summaries · {status.measurements ?? 0} measurements ·
            model {status.llm_reachable === false ? 'unreachable' : 'ready'}
          </p>
        )}
      </header>

      {turns.length === 0 && (
        <div className="examples">
          <p className="muted">Try one:</p>
          {EXAMPLES.map((e) => (
            <button key={e} className="example" onClick={(ev) => submit(ev, e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      <main>
        {turns.map((turn, i) => <Turn key={i} turn={turn} />)}
        {busy && (
          <p className="muted thinking">
            Working. Generation runs locally and takes 10 to 50 seconds.
          </p>
        )}
        <div ref={endRef} />
      </main>

      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. average surface temperature per year in the Arabian Sea"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}
