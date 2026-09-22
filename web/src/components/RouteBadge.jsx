// The route and who decided it. Shown on every turn rather than on request,
// because which half of the system answered is the first thing you need to
// know when an answer looks wrong.
export default function RouteBadge({ route, decidedBy, cached }) {
  if (!route) return null

  return (
    <div className="route">
      <span className={`badge badge-${route}`}>{route}</span>
      {decidedBy && <span className="muted">decided by {decidedBy}</span>}
      {cached && <span className="muted">· from cache</span>}
    </div>
  )
}
