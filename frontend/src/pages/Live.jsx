import { useEffect, useMemo, useState } from 'react'
import Hero from '../components/Hero'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'

const STALE_AFTER_HOURS = 48

function useLiveForecasts() {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(assetUrl('data/live_forecasts.json'))
        if (!response.ok) throw new Error('missing live forecast file')
        setState({ data: await response.json(), loading: false, error: null })
      } catch (err) {
        setState({ data: null, loading: false, error: 'Live forecasts are not available right now.' })
      }
    }
    load()
  }, [])

  return state
}

function hoursSince(iso) {
  return (Date.now() - new Date(iso).getTime()) / 36e5
}

function BandChart({ days, todayISO }) {
  const width = 860
  const height = 320
  const margin = { top: 16, right: 16, bottom: 42, left: 44 }
  const iw = width - margin.left - margin.right
  const ih = height - margin.top - margin.bottom

  const { xFor, yFor, yTicks, todayX } = useMemo(() => {
    const lo = Math.min(0, ...days.map(d => d.lo))
    const hi = Math.max(...days.map(d => d.hi))
    const pad = (hi - lo) * 0.08
    const yMin = lo - pad
    const yMax = hi + pad
    const xFor = i => margin.left + (i / Math.max(1, days.length - 1)) * iw
    const yFor = v => margin.top + ih - ((v - yMin) / (yMax - yMin)) * ih
    const step = (yMax - yMin) / 5
    const yTicks = Array.from({ length: 6 }, (_, i) => Math.round(yMin + i * step))
    const todayIdx = days.findIndex(d => d.date >= todayISO)
    const todayX = todayIdx >= 0 ? xFor(todayIdx) : null
    return { xFor, yFor, yTicks, todayX }
  }, [days, todayISO])

  const bandPath = useMemo(() => {
    const upper = days.map((d, i) => `${i === 0 ? 'M' : 'L'}${xFor(i)},${yFor(d.hi)}`).join('')
    const lower = days.slice().reverse().map((d, i) =>
      `L${xFor(days.length - 1 - i)},${yFor(d.lo)}`).join('')
    return `${upper}${lower}Z`
  }, [days, xFor, yFor])

  const q50Path = useMemo(() =>
    days.map((d, i) => `${i === 0 ? 'M' : 'L'}${xFor(i)},${yFor(d.q50)}`).join(''),
  [days, xFor, yFor])

  const labelEvery = Math.ceil(days.length / 8)

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }} role="img"
         aria-label="Live delay forecast with 80 percent interval band">
      {yTicks.map(t => (
        <g key={t}>
          <line x1={margin.left} x2={width - margin.right} y1={yFor(t)} y2={yFor(t)}
                stroke="#e3e3e3" strokeWidth="1" />
          <text x={margin.left - 8} y={yFor(t) + 4} textAnchor="end" fontSize="11" fill="#777">
            {t}
          </text>
        </g>
      ))}
      <path d={bandPath} fill="#4a7bd0" opacity="0.16" />
      <path d={q50Path} fill="none" stroke="#2c5cb8" strokeWidth="2" />
      {todayX !== null && (
        <g>
          <line x1={todayX} x2={todayX} y1={margin.top} y2={height - margin.bottom}
                stroke="#333" strokeWidth="1" strokeDasharray="5,4" />
          <text x={todayX + 6} y={margin.top + 12} fontSize="11" fill="#333">today</text>
        </g>
      )}
      <text x={margin.left + 4} y={margin.top + 12} fontSize="11" fill="#a04040">
        last actual → model feeds on its own predictions
      </text>
      {days.map((d, i) => (i % labelEvery === 0 ? (
        <text key={d.date} x={xFor(i)} y={height - margin.bottom + 18} textAnchor="middle"
              fontSize="11" fill="#777">
          {d.date.slice(5)}
        </text>
      ) : null))}
      <text x={14} y={margin.top + ih / 2} fontSize="11" fill="#777"
            transform={`rotate(-90 14 ${margin.top + ih / 2})`} textAnchor="middle">
        avg arrival delay (min)
      </text>
    </svg>
  )
}

function Live() {
  const { data, loading, error } = useLiveForecasts()
  const [route, setRoute] = useState(null)

  const routes = useMemo(() => (data && data.routes ? Object.keys(data.routes).sort() : []), [data])
  const selected = route || routes[0]
  const days = (data && selected && data.routes[selected]) || []
  const stale = data && hoursSince(data.generated_at) > STALE_AFTER_HOURS

  return (
    <>
      <Hero
        kicker="Live Forecasts"
        title="Seven Days Ahead, Honestly"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            BTS publishes flight actuals one to two months late, so a real forecaster cannot
            peek at yesterday. Every value on this page is rolled forward recursively from the
            last published actual: the model recomputes its own delay-lag features from its
            own predictions, day after day, with the same engine the recursion-depth backtest
            validated at every depth shown here. The backtest could not simulate everything:
            days beyond the weather archive use forecast weather, and the serving model is a
            newer generation than the backtested one — differences the model card discloses.
          </span>
          <span style={{ display: 'block' }}>
            The band targets 80% coverage, conformally widened per forecast depth.
            Wider bands further out are not a bug — they are the honest price of forecasting.
          </span>
        </>}
      />

      <section className="container" style={{ paddingBottom: 'var(--space-xl, 48px)' }}>
        {loading && <p>Loading live forecasts…</p>}
        {error && <p>{error}</p>}

        {data && (
          <>
            {stale && (
              <div style={{
                background: '#fff4e0', border: '1px solid #e0b060', borderRadius: 6,
                padding: '10px 14px', marginBottom: 16, fontSize: 14,
              }}>
                These forecasts were generated {Math.floor(hoursSince(data.generated_at))} hours
                ago and may be stale. The daily pipeline publishes a fresh set each morning;
                a gap here means a run failed and the last honest output was kept.
              </div>
            )}

            <div style={{ display: 'flex', gap: 16, alignItems: 'baseline', flexWrap: 'wrap',
                          marginBottom: 12 }}>
              <label htmlFor="route-select" style={{ fontSize: 14 }}>Route</label>
              <select id="route-select" value={selected} onChange={e => setRoute(e.target.value)}
                      style={{ fontSize: 14, padding: '4px 8px' }}>
                {routes.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
              <span style={{ fontSize: 13, color: '#777' }}>
                generated {new Date(data.generated_at).toLocaleString()} · last BTS
                actual {data.last_actual_date} · model {data.model_version} · trained
                through {data.trained_through}
              </span>
            </div>

            {days.length > 0 && <BandChart days={days} todayISO={data.today} />}

            <p style={{ fontSize: 13, color: '#777', marginTop: 10 }}>
              Forecast depth on this chart runs from k=1 (the day after the last published
              actual) to k={days.length ? days[days.length - 1].k : '—'}. Accuracy and
              interval calibration by depth are published in the repository's recursion
              backtest. Each day's run logs its published horizon to an append-only record
              that is never rewritten, so anyone can grade these forecasts against the
              actuals once BTS publishes them.
            </p>
          </>
        )}
      </section>

      <Footer />
    </>
  )
}

export default Live
