import { useEffect, useState } from 'react'
import Hero from '../components/Hero'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'

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
  const replay = replayStats(live)

  const totals = report && report.totals
  const graded = totals && totals.graded_rows > 0

  return (
    <>
      <Hero
        kicker="Monitoring"
        title="Forecasts, Graded in Public"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            Every morning's forecasts are appended to a log before any outcome is
            known, and past entries are never revised. When BTS publishes a month
            of actuals, everything logged for those dates gets scored here,
            exactly as issued.
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
          {!report && <p>The grading report is not available right now.</p>}
          {report && (
            <div className="viz-card" style={{ height: 'auto', padding: 'var(--space-lg)' }}>
              <div className="forecast-summary-metrics" style={{ maxWidth: 520 }}>
                <StatRow label="Forecasts logged" value={totals.logged_rows.toLocaleString()} />
                <StatRow label="Vintage days" value={totals.vintage_days} />
                <StatRow label="Logging since" value={totals.first_vintage || '-'} />
                <StatRow label="Graded against actuals" value={totals.graded_rows.toLocaleString()} />
                <StatRow label="Awaiting publication" value={totals.pending_rows.toLocaleString()} />
                <StatRow label="Actuals published through" value={report.actuals_through} />
              </div>
            </div>
          )}

          {report && !graded && (
            <p style={{ marginTop: 'var(--space-md)', maxWidth: 720 }}>
              Nothing is gradeable yet, and that is the expected state rather
              than a gap: logging began on {totals.first_vintage}, BTS actuals
              currently end at {report.actuals_through}, and the first logged
              dates become scorable when BTS publishes their month, one to two
              months after the fact. Until then the replay below shows what
              grading will look like.
            </p>
          )}

          {report && graded && (
            <div style={{ marginTop: 'var(--space-lg)' }}>
              <h3 style={{ marginBottom: 'var(--space-sm)' }}>Accuracy by forecast depth</h3>
              <div className="viz-card" style={{ height: 'auto', padding: 'var(--space-md)', overflowX: 'auto' }}>
                <DepthTable rows={report.by_depth} />
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: 'var(--space-sm)' }}>
                Scores are recomputed against the latest BTS revision on every
                run. The persistence column carries each vintage's last known
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
            The daily payload replays the last published month from the prior
            month's end with the same engine and offsets, so the grading
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
