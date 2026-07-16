import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { showTooltip, hideTooltip } from '../visualizations/tooltip.js'
import { extractCityPairs, buildDirectionalRoute, formatCityPairDisplay } from '../utils/routeUtils.js'
import DirectionToggle from './DirectionToggle.jsx'

const parseDate = d3.timeParse('%Y-%m-%d')
const formatAxisDate = d3.timeFormat('%b %d')
const formatTooltipDate = d3.timeFormat('%a, %b %d, %Y')
const formatMetaDate = d3.timeFormat('%b %d, %Y')
const formatMonthYear = d3.timeFormat('%B %Y')

const MODEL_DETAILS = {
  architecture: 'Three LightGBM quantile boosters (q10 / q50 / q90); the median feeds the recursion',
  calibration: 'Interval widening fit per forecast depth on a walk-forward recursion backtest, validated to 90 days',
  features: '80 features: calendar, route statistics, delay lags and rolling stats, daily/hourly weather, aviation weather, hub inbound state',
  serving: 'Regenerated daily at 09:30 UTC by a scheduled pipeline; a run with incomplete weather coverage publishes nothing'
}

function LiveForecastChart({ data, loading, error }) {
  const chartAreaRef = useRef(null)
  const mainChartRef = useRef(null)
  const [selectedCityPair, setSelectedCityPair] = useState(null)
  const [direction, setDirection] = useState('outbound')
  const [showMethodology, setShowMethodology] = useState(false)
  const [containerWidth, setContainerWidth] = useState(0)

  const routes = useMemo(
    () => (data && data.routes ? Object.keys(data.routes).sort() : []),
    [data]
  )

  const cityPairs = useMemo(() => extractCityPairs(routes), [routes])

  useEffect(() => {
    if (cityPairs.length > 0 && !selectedCityPair) {
      // a high-variance marquee route makes a better first impression than
      // the alphabetically first pair
      setSelectedCityPair(cityPairs.includes('DFW-LAX') ? 'DFW-LAX' : cityPairs[0])
    }
  }, [cityPairs])

  const selectedRoute = useMemo(() => {
    if (!selectedCityPair) return null
    const preferred = buildDirectionalRoute(selectedCityPair, direction)
    if (routes.includes(preferred)) return preferred
    return buildDirectionalRoute(selectedCityPair, direction === 'outbound' ? 'inbound' : 'outbound')
  }, [selectedCityPair, direction, routes])

  const series = useMemo(() => {
    if (!data || !selectedRoute || !data.routes[selectedRoute]) return null
    const { actuals, forecast, verification } = data.routes[selectedRoute]
    const actualByDate = new Map(actuals.map(a => [a.date, a.actual]))
    const actualRows = actuals.map(a => ({
      date: parseDate(a.date), actual: a.actual
    }))
    const forecastRows = forecast.map(f => ({
      date: parseDate(f.date), q50: f.q50, lo: f.lo, hi: f.hi, k: f.k
    }))
    const verRows = (verification || []).map(v => ({
      date: parseDate(v.date), q50: v.q50, lo: v.lo, hi: v.hi, k: v.k,
      actual: actualByDate.get(v.date) ?? null
    }))
    return {
      all: [...actualRows, ...forecastRows].sort((a, b) => a.date - b.date),
      forecastRows,
      verRows,
      verByTime: new Map(verRows.map(v => [+v.date, v])),
      lastActual: parseDate(data.last_actual_date),
      today: parseDate(data.today)
    }
  }, [data, selectedRoute])

  const summary = useMemo(() => {
    if (!series) return null
    const upcoming = series.forecastRows.filter(d => d.date >= series.today)
    if (upcoming.length === 0) return null
    const horizon = upcoming[upcoming.length - 1]
    const graded = series.verRows.filter(v => v.actual != null)
    return {
      outlookMean: d3.mean(upcoming, d => d.q50),
      horizon,
      kMin: upcoming[0].k,
      kMax: horizon.k,
      verMae: graded.length ? d3.mean(graded, v => Math.abs(v.q50 - v.actual)) : null,
      verInBand: graded.filter(v => v.actual >= v.lo && v.actual <= v.hi).length,
      verDays: graded.length
    }
  }, [series])

  useEffect(() => {
    const el = chartAreaRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = Math.round(entries[0].contentRect.width)
      setContainerWidth(prev => (prev !== w ? w : prev))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [loading])

  useEffect(() => {
    if (!series || !mainChartRef.current) return

    const container = mainChartRef.current
    container.innerHTML = ''

    const margin = { top: 20, right: 30, bottom: 30, left: 55 }
    const width = container.clientWidth - margin.left - margin.right
    const height = 320 - margin.top - margin.bottom
    if (width <= 0) return

    const { all, forecastRows, verRows, verByTime, lastActual, today } = series

    const xScale = d3.scaleTime()
      .domain(d3.extent(all, d => d.date))
      .range([0, width])

    const domainRows = [...all, ...verRows]
    const yMin = d3.min(domainRows, d => Math.min(d.actual ?? Infinity, d.lo ?? Infinity))
    const yMax = d3.max(domainRows, d => Math.max(d.actual ?? -Infinity, d.hi ?? -Infinity))
    const pad = (yMax - yMin) * 0.1
    const yScale = d3.scaleLinear()
      .domain([Math.min(0, yMin - pad), yMax + pad])
      .range([height, 0])

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).ticks(6).tickSize(-width).tickFormat(''))
      .selectAll('line')
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-dasharray', '2,2')
    g.selectAll('.grid .domain').remove()

    const boundaryX = xScale(lastActual)

    g.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', Math.max(0, boundaryX))
      .attr('height', height)
      .attr('fill', 'var(--bg-base-soft)')
      .attr('opacity', 0.3)

    const band = d3.area()
      .defined(d => d.lo != null)
      .x(d => xScale(d.date))
      .y0(d => yScale(d.lo))
      .y1(d => yScale(d.hi))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(forecastRows)
      .attr('fill', 'var(--blue)')
      .attr('opacity', 0.13)
      .attr('d', band)

    g.append('path')
      .datum(verRows)
      .attr('fill', 'var(--blue)')
      .attr('opacity', 0.13)
      .attr('d', band)

    if (yScale.domain()[0] < 0) {
      g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', 'var(--text-muted)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4')
        .attr('opacity', 0.5)
    }

    g.append('line')
      .attr('x1', boundaryX)
      .attr('x2', boundaryX)
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', 'var(--red)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')

    const todayX = xScale(today)
    g.append('line')
      .attr('x1', todayX)
      .attr('x2', todayX)
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.7)
    g.append('text')
      .attr('x', todayX + 5)
      .attr('y', 12)
      .attr('font-size', '11px')
      .attr('fill', 'var(--text-muted)')
      .text('Today')

    const lineActual = d3.line()
      .defined(d => d.actual != null)
      .x(d => xScale(d.date))
      .y(d => yScale(d.actual))
      .curve(d3.curveMonotoneX)

    const lineForecast = d3.line()
      .defined(d => d.q50 != null)
      .x(d => xScale(d.date))
      .y(d => yScale(d.q50))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(all)
      .attr('fill', 'none')
      .attr('stroke', 'var(--green)')
      .attr('stroke-width', 1.5)
      .attr('d', lineActual)

    g.append('path')
      .datum(forecastRows)
      .attr('fill', 'none')
      .attr('stroke', 'var(--blue)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,2')
      .attr('d', lineForecast)

    g.append('path')
      .datum(verRows)
      .attr('fill', 'none')
      .attr('stroke', 'var(--blue)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,2')
      .attr('d', lineForecast)

    const bisect = d3.bisector(d => d.date).center
    const focusLine = g.append('line')
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-dasharray', '3,3')
      .attr('y1', 0)
      .attr('y2', height)
      .style('display', 'none')
    const focusActual = g.append('circle')
      .attr('r', 4).attr('fill', 'var(--green)')
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .style('display', 'none')
    const focusForecast = g.append('circle')
      .attr('r', 4).attr('fill', 'var(--blue)')
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .style('display', 'none')

    g.append('rect')
      .attr('fill', 'transparent')
      .attr('width', width)
      .attr('height', height)
      .style('cursor', 'crosshair')
      .on('mousemove', (event) => {
        const [mx] = d3.pointer(event)
        const date = xScale.invert(mx)
        const i = bisect(all, date)
        const d = all[Math.min(i, all.length - 1)]
        if (!d) return

        focusLine.style('display', null)
          .attr('x1', xScale(d.date)).attr('x2', xScale(d.date))

        const ver = d.actual != null ? verByTime.get(+d.date) : null

        if (d.actual != null) {
          focusActual.style('display', null)
            .attr('cx', xScale(d.date)).attr('cy', yScale(d.actual))
        } else {
          focusActual.style('display', 'none')
        }
        const forecastValue = d.q50 ?? ver?.q50
        if (forecastValue != null) {
          focusForecast.style('display', null)
            .attr('cx', xScale(d.date)).attr('cy', yScale(forecastValue))
        } else {
          focusForecast.style('display', 'none')
        }

        let html = `<div><strong>${formatTooltipDate(d.date)}</strong></div>`
        if (d.actual != null) {
          html += `<div style="color: var(--green);">${d.actual.toFixed(1)} min actual</div>`
        }
        if (ver) {
          const err = ver.q50 - d.actual
          html += `<div style="color: var(--blue);">${ver.q50.toFixed(1)} min forecast</div>`
          html += `<div>Error: ${err >= 0 ? '+' : ''}${err.toFixed(1)} min</div>`
          html += `<div style="color: var(--text-muted); font-size: 11px;">Replay, ${ver.k} day${ver.k === 1 ? '' : 's'} out</div>`
        } else if (d.actual != null) {
          html += `<div style="color: var(--text-muted); font-size: 11px;">Observed (BTS)</div>`
        }
        if (d.q50 != null) {
          html += `<div style="color: var(--blue);">${d.q50.toFixed(1)} min forecast</div>`
          html += `<div>80% interval: ${d.lo.toFixed(1)} to ${d.hi.toFixed(1)} min</div>`
          html += `<div style="color: var(--text-muted); font-size: 11px;">${d.k} day${d.k === 1 ? '' : 's'} past last actual</div>`
        }
        showTooltip(event, html, { bounds: container })
      })
      .on('mouseleave', () => {
        focusLine.style('display', 'none')
        focusActual.style('display', 'none')
        focusForecast.style('display', 'none')
        hideTooltip()
      })

    const xAxisG = g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(7).tickFormat(formatAxisDate))
    xAxisG.selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '11px')

    const yAxisG = g.append('g')
      .call(d3.axisLeft(yScale).ticks(6))
    yAxisG.selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '11px')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', 'var(--text-muted)')
      .text('Avg arrival delay (min)')

  }, [series, containerWidth])

  if (loading) {
    return (
      <section id="live-forecast" className="section section--alt">
        <div className="container">
          <p className="kicker">Daily Forecast</p>
          <h2>Seven-Day Route Outlook</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading live forecasts...
          </div>
        </div>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section id="live-forecast" className="section section--alt">
        <div className="container">
          <p className="kicker">Daily Forecast</p>
          <h2>Seven-Day Route Outlook</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error || 'Live forecasts are not available right now.'}
          </div>
        </div>
      </section>
    )
  }

  const lastActualDate = parseDate(data.last_actual_date)
  const horizonDate = parseDate(data.horizon_end)
  const todayDate = parseDate(data.today)
  const generatedAt = new Date(data.generated_at)
  const hoursSinceUpdate = (Date.now() - generatedAt.getTime()) / 36e5
  const stale = hoursSinceUpdate > 48

  return (
    <section id="live-forecast" className="section section--alt">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Daily Forecast</p>
        <h2>Seven-Day Route Outlook</h2>
        <p style={{ marginBottom: 'var(--space-sm)' }}>
          Observed average arrival delay is shown in green through {formatMetaDate(lastActualDate)},
          the most recent month published by the Bureau of Transportation Statistics. Forecasts
          continue in dashed blue through {formatMetaDate(horizonDate)}, with the shaded band
          covering the 80% prediction interval.
        </p>
        <p style={{ marginBottom: 'var(--space-md)' }}>
          The red line marks the last published actual. To the right of it the model has no
          observed delay history to draw on, so its lag features are rolled forward from its own
          prior predictions one day at a time, and days beyond the weather archive run on
          forecast weather rather than observations. Left of the line, the dashed overlay
          replays the last published month from
          a {data ? formatMetaDate(parseDate(data.verification_origin)) : 'month-end'} start
          with the same engine and offsets. Its delay features roll forward from the model's own
          predictions exactly as they do live, but its weather comes from the settled archive,
          which a true month-end run would not have had. The replay isolates what recursive
          feedback costs against known outcomes, scored in the panel on the right, and reads as
          a favorable bound for the forward forecast.
        </p>

        <div className="methodology-card" style={{ marginBottom: 'var(--space-lg)' }}>
          <button
            className="methodology-toggle"
            onClick={() => setShowMethodology(!showMethodology)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 0',
              color: 'var(--text-primary)', fontSize: '13px', fontWeight: '600'
            }}
          >
            <span style={{ transform: showMethodology ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>&#9654;</span>
            Model Details: LightGBM Quantile Ensemble
            <span className="model-detail-badge model-detail-badge--gradient-boosting">
              Gradient Boosting
            </span>
          </button>
          {showMethodology && (
            <div className="model-details-grid">
              <div className="model-detail-item">
                <span className="model-detail-label">Architecture</span>
                <span className="model-detail-value">{MODEL_DETAILS.architecture}</span>
              </div>
              <div className="model-detail-item">
                <span className="model-detail-label">Training</span>
                <span className="model-detail-value">
                  Trained through {formatMetaDate(parseDate(data.trained_through))}; early
                  stopping and conformal calibration on disjoint halves of Jan-May 2026
                </span>
              </div>
              <div className="model-detail-item">
                <span className="model-detail-label">Calibration</span>
                <span className="model-detail-value">{MODEL_DETAILS.calibration}</span>
              </div>
              <div className="model-detail-item">
                <span className="model-detail-label">Features</span>
                <span className="model-detail-value">{MODEL_DETAILS.features}</span>
              </div>
              <div className="model-detail-item">
                <span className="model-detail-label">Serving</span>
                <span className="model-detail-value">{MODEL_DETAILS.serving}</span>
              </div>
            </div>
          )}
        </div>

        {stale && (
          <div style={{
            background: 'var(--bg-base-soft)', border: '1px solid var(--red)',
            borderRadius: '6px', padding: '10px 14px',
            marginBottom: 'var(--space-md)', fontSize: '13px'
          }}>
            Forecasts were last updated {Math.floor(hoursSinceUpdate)} hours ago. The daily
            refresh has not published since; the values below are the most recent available.
          </div>
        )}

        <div className="viz-card" style={{ height: 'auto', padding: 0 }}>
          <div className="forecast-layout">
            <div className="forecast-controls">
              <h4>Controls</h4>
              <label>
                Route
                <select
                  className="select"
                  value={selectedCityPair || ''}
                  onChange={(e) => setSelectedCityPair(e.target.value)}
                >
                  {cityPairs.map(pair => (
                    <option key={pair} value={pair}>{formatCityPairDisplay(pair)}</option>
                  ))}
                </select>
                <DirectionToggle
                  cityPair={selectedCityPair}
                  direction={direction}
                  onChange={setDirection}
                  availableRoutes={routes}
                />
              </label>

              <div className="forecast-legend" style={{ marginTop: 'var(--space-md)' }}>
                <div className="legend-item">
                  <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="var(--green)" strokeWidth="2"/></svg>
                  <span>Actual</span>
                </div>
                <div className="legend-item">
                  <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="var(--blue)" strokeWidth="2" strokeDasharray="4,2"/></svg>
                  <span>Forecast</span>
                </div>
                <div className="legend-item">
                  <svg width="20" height="10"><rect x="0" y="1" width="20" height="8" fill="var(--blue)" opacity="0.13"/></svg>
                  <span>80% Interval</span>
                </div>
                <div className="legend-item">
                  <svg width="20" height="10"><line x1="10" y1="0" x2="10" y2="10" stroke="var(--red)" strokeWidth="2" strokeDasharray="3,2"/></svg>
                  <span>Last Actual</span>
                </div>
                <div className="legend-item">
                  <svg width="20" height="10"><line x1="10" y1="0" x2="10" y2="10" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="2,2"/></svg>
                  <span>Today</span>
                </div>
              </div>
            </div>

            <div className="forecast-chart" ref={chartAreaRef}>
              <div ref={mainChartRef} style={{ width: '100%', height: '320px' }} />
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 'var(--space-xs)' }}>
                Hover the chart for daily values.
              </p>
            </div>

            <div className="forecast-summary">
              <div className="forecast-summary__header">
                <span className="forecast-summary__period">
                  Forecast window, {formatAxisDate(todayDate)}-{formatAxisDate(horizonDate)}
                </span>
                <div className="summary-header">
                  <span className="summary-header__model">LightGBM median</span>
                  <span className="summary-header__route">{selectedRoute}</span>
                </div>
              </div>

              <div className="forecast-summary__metrics-grid">
                <div className="forecast-summary__column">
                  <h4>Through {formatAxisDate(horizonDate)}</h4>
                  <div className="forecast-summary-metrics">
                    <div className="metric-row">
                      <span className="metric-row__label">Outlook mean</span>
                      <span className="metric-row__value">
                        {summary ? `${summary.outlookMean.toFixed(1)} min` : '-'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">Band, {summary ? formatAxisDate(summary.horizon.date) : '-'}</span>
                      <span className="metric-row__value">
                        {summary ? `${summary.horizon.lo.toFixed(0)} to ${summary.horizon.hi.toFixed(0)} min` : '-'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">Days past last actual</span>
                      <span className="metric-row__value">
                        {summary ? `${summary.kMin}-${summary.kMax}` : '-'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="forecast-summary__column">
                  <h4>{formatMonthYear(lastActualDate)} Replay</h4>
                  <div className="forecast-summary-metrics">
                    <div className="metric-row">
                      <span className="metric-row__label">Forecast MAE</span>
                      <span className="metric-row__value">
                        {summary && summary.verMae != null ? `${summary.verMae.toFixed(1)} min` : '-'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">Inside 80% band</span>
                      <span className="metric-row__value">
                        {summary && summary.verDays ? `${summary.verInBand} of ${summary.verDays} days` : '-'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="forecast-summary__column">
                  <h4>This Run</h4>
                  <div className="forecast-summary-metrics">
                    <div className="metric-row">
                      <span className="metric-row__label">Updated</span>
                      <span className="metric-row__value">{formatMetaDate(generatedAt)}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">Actuals through</span>
                      <span className="metric-row__value">{formatMetaDate(lastActualDate)}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">Trained through</span>
                      <span className="metric-row__value">
                        {formatMetaDate(parseDate(data.trained_through))}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: 'var(--space-md)' }}>
          Each morning's forecasts are appended to a versioned log before any outcome is known,
          and past entries are never revised. Once BTS publishes a month of actuals, every
          forecast for those dates can be scored against them.
        </p>
      </div>
    </section>
  )
}

export default LiveForecastChart
