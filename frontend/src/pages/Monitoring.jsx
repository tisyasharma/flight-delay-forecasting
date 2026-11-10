import { useEffect, useMemo, useState } from 'react'
import Hero from '../components/Hero'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'
import { isStale, useMinuteClock } from '../utils/freshness.js'

const INCIDENT_URL =
  'https://github.com/tisyasharma/flight-delay-forecasting/blob/main/docs/incidents/2024-12-openmeteo-wind-seam.md'

function useJson(path) {
  const [data, setData] = useState(null)
  useEffect(() => {
    let alive = true
    fetch(assetUrl(path))
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (alive) setData(d) })
      .catch(() => { if (alive) setData(null) })
    return () => { alive = false }
  }, [path])
  return data
}

function replayStats(live) {
  if (!live || !live.routes) return null
  let absErr = 0, inBand = 0, n = 0
  Object.values(live.routes).forEach(({ actuals, verification }) => {
    const actualByDate = new Map(actuals.map(a => [a.date, a.actual]))
    ;(verification || []).forEach(v => {
      const actual = actualByDate.get(v.date)
      if (actual == null) return
      absErr += Math.abs(v.q50 - actual)
      if (actual >= v.lo && actual <= v.hi) inBand += 1
      n += 1
    })
  })
  if (!n) return null
  return { mae: absErr / n, inBand, n, coverage: (100 * inBand) / n }
}

function StatRow({ label, value }) {
  return (
    <div className="metric-row">
      <span className="metric-row__label">{label}</span>
      <span className="metric-row__value">{value}</span>
    </div>
  )
}

function DepthTable({ rows }) {
  return (
    <table className="data-table" style={{ width: '100%', fontSize: '13px' }}>
      <thead>
        <tr>
          <th>Days past last actual</th><th>Forecasts</th><th>MAE</th>
          <th>Persistence MAE</th><th>Coverage (80% target)</th><th>WIS</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.k}>
            <td>{r.k}</td><td>{r.n}</td><td>{r.mae} min</td>
            <td>{r.persistence_mae != null ? `${r.persistence_mae} min` : '-'}</td>
            <td>{r.coverage_80}%</td><td>{r.wis}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Monitoring() {
  const report = useJson('data/monitoring.json')
  const live = useJson('data/live_forecasts.json')
  const replay = useMemo(() => replayStats(live), [live])

  const totals = report && report.totals
  const graded = totals && totals.graded_rows > 0
  const nowMs = useMinuteClock()
  // grading only runs inside the daily job, so a grader timestamp this old means the job stopped short of it
  const gradingPaused = Boolean(report && report.graded_at) &&
    isStale(new Date(report.graded_at), nowMs)

  return (
    <>
      <Hero
        kicker="Monitoring"
        title="Forecast Log and Grading"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            Each published run appends its forecasts to a log before any outcome
            is known, and past entries are never revised. Each successful run of
            the daily job grades everything logged for dates whose BTS actuals are
            in the repo, exactly as issued.
            {totals && totals.last_vintage && <> The most recent run logged is {totals.last_vintage}.</>}
          </span>
          <span style={{ display: 'block' }}>
            The wait between forecast and verdict is the project's defining
            constraint, so this page reports the pending count as plainly as the
            scores.
          </span>
        </>}
      />

      <section className="section section--alt">
        <div className="container">
          <p className="kicker">The Record</p>
          <h2>What the log holds</h2>
          {!totals && <p>The grading report is not available right now.</p>}
          {totals && (
            <div className="viz-card" style={{ height: 'auto', padding: 'var(--space-lg)' }}>
              <div className="forecast-summary-metrics" style={{ maxWidth: 520 }}>
                <StatRow label="Forecasts logged" value={totals.logged_rows.toLocaleString()} />
                <StatRow label="Vintage days" value={totals.vintage_days} />
                <StatRow label="Logging since" value={totals.first_vintage || '-'} />
                <StatRow label="Graded against actuals" value={totals.graded_rows.toLocaleString()} />
                <StatRow label="Awaiting actuals" value={totals.pending_rows.toLocaleString()} />
                <StatRow label="Actuals through" value={report.actuals_through} />
                <StatRow label="Grader last ran" value={report.graded_at ? report.graded_at.slice(0, 10) : '-'} />
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: 'var(--space-sm)' }}>
                All figures are as of the grader's last run.
                {gradingPaused && ' Grading is a step of the daily job, and no run has reached that step since then.'}
              </p>
            </div>
          )}

          {totals && !graded && totals.first_vintage && report.actuals_through && (
            <p style={{ marginTop: 'var(--space-md)', maxWidth: 720 }}>
              Nothing has been graded yet. Logging began on {totals.first_vintage},
              and when the grader last ran the actuals ended at {report.actuals_through}
              {report.actuals_through < totals.first_vintage ? ', before the first logged date' : ''}.
              Scoring needs BTS actuals for the logged
              dates in the repo and a successful run of the daily job, whose grading
              step does the scoring.
              {replay && ' Until then, the replay below shows what grading will look like.'}
            </p>
          )}

          {report && graded && (
            <div style={{ marginTop: 'var(--space-lg)' }}>
              <h3 style={{ marginBottom: 'var(--space-sm)' }}>Accuracy by forecast depth</h3>
              <div className="viz-card" style={{ height: 'auto', padding: 'var(--space-md)', overflowX: 'auto' }}>
                <DepthTable rows={report.by_depth} />
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: 'var(--space-sm)' }}>
                Scores are recomputed against the BTS actuals in the repo on
                every successful run. The persistence column carries each vintage's last known
                actual forward, the same comparator the recursion backtest
                reports.
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="section">
        <div className="container">
          <p className="kicker">Proof of Concept</p>
          <h2>What grading looks like</h2>
          <p style={{ maxWidth: 720 }}>
            The payload from the last published run replays its last month of
            actuals from the prior month's end with the same engine and offsets, so the grading
            machinery already runs against known outcomes. Its weather comes
            from the settled archive, which a true month-end run would not have
            had, so these numbers read as a favorable bound.
          </p>
          {replay && (
            <div className="viz-card" style={{ height: 'auto', padding: 'var(--space-lg)', maxWidth: 520 }}>
              <div className="forecast-summary-metrics">
                <StatRow label="Replay forecasts scored" value={replay.n.toLocaleString()} />
                <StatRow label="Replay MAE" value={`${replay.mae.toFixed(1)} min`} />
                <StatRow
                  label="Inside the 80% band"
                  value={`${replay.inBand.toLocaleString()} of ${replay.n.toLocaleString()} (${replay.coverage.toFixed(1)}%)`}
                />
              </div>
            </div>
          )}
          <p style={{ maxWidth: 720, marginTop: 'var(--space-md)' }}>
            Monitoring here means catching problems, not painting green walls.
            The recursion backtest caught raw interval coverage decaying from
            74% to 62% as the model feeds on its own predictions, which is why
            every depth carries its own widening. The data-quality gates caught
            an upstream weather source silently changing models mid-series,
            written up in the{' '}
            <a href={INCIDENT_URL}>wind-seam incident note</a>, and the response
            was to fix the source rather than retrain on contaminated data.
          </p>
        </div>
      </section>

      <Footer />
    </>
  )
}

export default Monitoring
