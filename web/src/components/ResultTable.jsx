const MAX_ROWS = 100

export default function ResultTable({ columns, rows, rowCount }) {
  if (!columns || !rows || rows.length === 0) return null

  const shown = rows.slice(0, MAX_ROWS)

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell === null ? <span className="muted">null</span> : String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rowCount > shown.length && (
        <p className="muted">
          showing {shown.length} of {rowCount} rows. Export to get all of them.
        </p>
      )}
    </div>
  )
}
