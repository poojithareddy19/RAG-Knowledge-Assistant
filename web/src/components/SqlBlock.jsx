// The query that produced the table, shown before the user is asked to
// believe the numbers. Collapsed by default so it does not shout, open in one
// click because an unverifiable table is the failure this project is built
// against.
import { useState } from 'react'

export default function SqlBlock({ sql }) {
  const [open, setOpen] = useState(false)

  if (!sql) return null

  return (
    <div className="sql">
      <button className="disclose" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} SQL that produced this
      </button>
      {open && <pre>{sql}</pre>}
    </div>
  )
}
