import Chart from './Chart.jsx'
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

          <Chart
            spec={result.chart_spec}
            png={result.chart_png_base64}
            kind={result.chart_kind}
          />
          <SqlBlock sql={result.generated_sql} />
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
