import { useEffect, useRef, useState } from 'react'

// Plotly is 4.8 MB minified, and most answers carry no chart, so it is loaded
// the first time a chart is shown rather than with the page.
let plotly = null

function loadPlotly() {
  plotly = plotly || import('plotly.js-dist-min').then((m) => m.default || m)
  return plotly
}

// The world basemaps for trajectory maps are served from public/topojson/
// rather than fetched from Plotly's CDN, so a map draws offline.
const CONFIG = {
  responsive: true,
  displaylogo: false,
  topojsonURL: '/topojson/',
}

function PlotlyFigure({ spec }) {
  const node = useRef(null)
  const [failed, setFailed] = useState(null)

  useEffect(() => {
    let cancelled = false
    const el = node.current

    loadPlotly()
      .then((Plotly) => {
        if (!cancelled && el) {
          return Plotly.react(el, spec.data || [], spec.layout || {}, CONFIG)
        }
      })
      .catch((err) => !cancelled && setFailed(String(err)))

    return () => {
      cancelled = true
      if (plotly && el) plotly.then((Plotly) => Plotly.purge(el))
    }
  }, [spec])

  // A chart that cannot be drawn costs the user the chart, not the answer.
  if (failed) return <p className="muted">The chart could not be drawn: {failed}</p>

  return <div ref={node} className="chart" />
}

// One of two things comes back: a Plotly figure when the rows are an ocean
// shape (a trajectory, profiles, a section, a T-S diagram), or a PNG from the
// generic renderer when a chart was asked for and the rows are an ordinary
// aggregate. The figure wins when both are present, as it always has.
export default function Chart({ spec, png, kind }) {
  if (spec) return <PlotlyFigure spec={spec} />

  if (png) {
    return (
      <img
        className="chart"
        src={`data:image/png;base64,${png}`}
        alt={kind ? `${kind} chart` : 'chart'}
      />
    )
  }

  return null
}
