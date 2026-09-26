import Citations from './Citations.jsx'
import ResultTable from './ResultTable.jsx'
import RouteBadge from './RouteBadge.jsx'
import SqlBlock from './SqlBlock.jsx'

export default function Turn({ turn }) {
  const { question, result, error } = turn

  return (
    <article className="turn">
      <p className="question">{question}</p>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <RouteBadge
            route={result.route}
            decidedBy={result.route_decided_by}
            cached={result.sql_cached}
          />

          {/* A refusal is a result, not an error. It is styled as an answer
              with its reason attached, because the system declining is the
              behaviour this project wants and hiding it as a failure would
              teach users to distrust it. */}
          {result.refused ? (
            <div className="refusal">
              <strong>Not answered.</strong> {result.reason || result.answer}
            </div>
          ) : (
            <p className="answer">{result.answer}</p>
          )}

          {result.question_rewritten && result.question_rewritten !== question && (
            <p className="muted">
              answered as: {result.question_rewritten}
            </p>
          )}

          <SqlBlock sql={result.generated_sql} />

          {/* A fixed MCP tool answered instead of generated SQL. What a reader
              checks then is the tool and its arguments, not a query. */}
          {result.mcp_tool && (
            <p className="muted">
              answered by MCP tool <code>{result.mcp_tool}</code>
              {result.mcp_arguments && ` with ${JSON.stringify(result.mcp_arguments)}`}
            </p>
          )}
          <ResultTable
            columns={result.columns}
            rows={result.rows}
            rowCount={result.row_count}
          />
          <Citations citations={result.citations} />

          <p className="meta muted">
            confidence {(result.confidence ?? 0).toFixed(2)}
            {result.elapsed_ms != null && ` · ${Math.round(result.elapsed_ms)} ms`}
            {result.row_count != null && ` · ${result.row_count} rows`}
          </p>
        </>
      )}
    </article>
  )
}
