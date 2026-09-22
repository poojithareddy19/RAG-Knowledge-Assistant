// Document and page behind every claim. A manual answer without these is the
// thing the confidence gate exists to prevent, so they are not collapsed.
export default function Citations({ citations }) {
  if (!citations || citations.length === 0) return null

  return (
    <div className="citations">
      <h4>Sources</h4>
      <ol>
        {citations.map((c, i) => (
          <li key={i}>
            <span className="doc">{c.doc_name ?? c.document ?? 'source'}</span>
            {c.page != null && <span className="muted"> · page {c.page}</span>}
            {c.score != null && <span className="muted"> · similarity {Number(c.score).toFixed(3)}</span>}
            {c.text && <p className="passage">{c.text}</p>}
          </li>
        ))}
      </ol>
    </div>
  )
}
