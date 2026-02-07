import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { assetUrl } from '../utils/helpers.js'


const delayColorScale = d3.scaleLinear()
  .domain([-5, 6, 18])
  .range(['#4f6d9a', '#7c5295', '#a94467'])
  .clamp(true)

const FEATURE_LABELS = {
  Precip: 'Precipitation',
  Severity: 'Severity Index',
  Snowfall: 'Snowfall',
  Wind: 'Wind Speed',
  Temp: 'Temperature'
}

function WeatherImpactChart() {
  const divergingRef = useRef(null)
  const lollipopRef = useRef(null)
  const chartWrapRef = useRef(null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState('impact')
  const [containerWidth, setContainerWidth] = useState(0)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      const response = await fetch(assetUrl('data/weather_impact.json'))
      if (!response.ok) throw new Error('Failed to load weather impact data')
      const jsonData = await response.json()
      setData(jsonData)
    } catch (err) {
      setError('Failed to load weather impact data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const el = chartWrapRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = Math.round(entries[0].contentRect.width)
      setContainerWidth(prev => prev !== w ? w : prev)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [data])

  useEffect(() => {
    if (!data || !divergingRef.current || viewMode !== 'impact') return
    drawDivergingChart()
  }, [data, viewMode, containerWidth])

  useEffect(() => {
    if (!data || !lollipopRef.current || viewMode !== 'correlation') return
    drawLollipopChart()
  }, [data, viewMode, containerWidth])

  const drawDivergingChart = () => {
    const container = d3.select(divergingRef.current)
    container.selectAll('*').remove()

    if (!data?.by_severity) return

    const sorted = [...data.by_severity].sort((a, b) => a.avg_delay - b.avg_delay)
    const containerWidth = divergingRef.current.clientWidth
    const rowHeight = 52
    const containerHeight = sorted.length * rowHeight + 70

    const margin = { top: 24, right: 120, bottom: 36, left: 110 }
    const chartWidth = containerWidth - margin.left - margin.right
    const chartHeight = sorted.length * rowHeight

    const svg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    const maxAbs = Math.max(...sorted.map(d => Math.abs(d.avg_delay))) * 1.15
    const xScale = d3.scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([0, chartWidth])

    const yScale = d3.scaleBand()
      .domain(sorted.map(d => d.label))
      .range([0, chartHeight])
      .padding(0.3)

    const zeroX = xScale(0)

    // grid lines
    const ticks = xScale.ticks(6)
    g.selectAll('.grid-line')
      .data(ticks)
      .enter()
      .append('line')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', 0)
      .attr('y2', chartHeight)
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-dasharray', '2,3')
      .attr('opacity', d => d === 0 ? 0 : 0.6)

    // zero line
    g.append('line')
      .attr('x1', zeroX)
      .attr('x2', zeroX)
      .attr('y1', -4)
      .attr('y2', chartHeight)
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-width', 1.5)

    // bars
    sorted.forEach((d, i) => {
      const barStart = d.avg_delay < 0 ? xScale(d.avg_delay) : zeroX
      const barWidth = Math.abs(xScale(d.avg_delay) - zeroX)
      const barColor = delayColorScale(d.avg_delay)

      g.append('rect')
        .attr('x', barStart)
        .attr('y', yScale(d.label))
        .attr('width', 0)
        .attr('height', yScale.bandwidth())
        .attr('fill', barColor)
        .attr('rx', 3)
        .attr('opacity', 0.88)
        .transition()
        .duration(500)
        .delay(i * 70)
        .attr('width', barWidth)

      // value label at the end of each bar
      const labelX = d.avg_delay < 0
        ? xScale(d.avg_delay) - 6
        : xScale(d.avg_delay) + 6
      const anchor = d.avg_delay < 0 ? 'end' : 'start'

      g.append('text')
        .attr('x', labelX)
        .attr('y', yScale(d.label) + yScale.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', anchor)
        .attr('fill', barColor)
        .attr('font-size', '12px')
        .attr('font-weight', '700')
        .attr('font-variant-numeric', 'tabular-nums')
        .attr('opacity', 0)
        .text(`${d.avg_delay > 0 ? '+' : ''}${d.avg_delay.toFixed(1)}`)
        .transition()
        .duration(300)
        .delay(i * 70 + 400)
        .attr('opacity', 1)
    })

    // y-axis labels
    g.append('g')
      .call(d3.axisLeft(yScale).tickSize(0))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '13px')
      .attr('font-weight', '500')

    // secondary info on the right: n_flights and high_delay_pct
    sorted.forEach((d, i) => {
      const y = yScale(d.label) + yScale.bandwidth() / 2

      g.append('text')
        .attr('x', chartWidth + 12)
        .attr('y', y - 7)
        .attr('fill', 'var(--text-muted)')
        .attr('font-size', '11px')
        .attr('font-variant-numeric', 'tabular-nums')
        .attr('opacity', 0)
        .text(`${d.n_flights.toLocaleString()} flights`)
        .transition()
        .duration(300)
        .delay(i * 70 + 300)
        .attr('opacity', 1)

      g.append('text')
        .attr('x', chartWidth + 12)
        .attr('y', y + 8)
        .attr('fill', d.high_delay_pct > 0.2 ? '#dc2626' : 'var(--text-muted)')
        .attr('font-size', '11px')
        .attr('font-weight', d.high_delay_pct > 0.2 ? '600' : '400')
        .attr('font-variant-numeric', 'tabular-nums')
        .attr('opacity', 0)
        .text(`${(d.high_delay_pct * 100).toFixed(1)}% over 15 min`)
        .transition()
        .duration(300)
        .delay(i * 70 + 300)
        .attr('opacity', 1)
    })

    // x-axis
    g.append('g')
      .attr('transform', `translate(0,${chartHeight + 4})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d => {
        if (d === 0) return '0'
        return `${d > 0 ? '+' : ''}${d.toFixed(0)}`
      }))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px')

    svg.append('text')
      .attr('x', margin.left + chartWidth / 2)
      .attr('y', containerHeight - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Average Delay (minutes)')
  }

  const drawLollipopChart = () => {
    const container = d3.select(lollipopRef.current)
    container.selectAll('*').remove()

    if (!data?.correlation) return

    const correlations = Object.entries(data.correlation).map(([key, value]) => ({
      feature: key.charAt(0).toUpperCase() + key.slice(1),
      correlation: value
    })).sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))

    const containerWidth = lollipopRef.current.clientWidth
    const rowHeight = 48
    const containerHeight = correlations.length * rowHeight + 70

    const margin = { top: 24, right: 60, bottom: 36, left: 110 }
    const chartWidth = containerWidth - margin.left - margin.right
    const chartHeight = correlations.length * rowHeight

    const svg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    const xScale = d3.scaleLinear()
      .domain([0, Math.max(0.35, d3.max(correlations, d => d.correlation) * 1.1)])
      .range([0, chartWidth])

    const yScale = d3.scaleBand()
      .domain(correlations.map(d => d.feature))
      .range([0, chartHeight])
      .padding(0.35)

    // grid
    const ticks = xScale.ticks(5)
    g.selectAll('.grid-line')
      .data(ticks.filter(t => t > 0))
      .enter()
      .append('line')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', 0)
      .attr('y2', chartHeight)
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-dasharray', '2,3')
      .attr('opacity', 0.6)

    // lollipop stems + dots
    correlations.forEach((d, i) => {
      const y = yScale(d.feature) + yScale.bandwidth() / 2
      const endX = xScale(Math.abs(d.correlation))

      // stem line
      g.append('line')
        .attr('x1', 0)
        .attr('y1', y)
        .attr('x2', 0)
        .attr('y2', y)
        .attr('stroke', '#1e40af')
        .attr('stroke-width', 2)
        .attr('opacity', 0.4)
        .transition()
        .duration(500)
        .delay(i * 80)
        .attr('x2', endX)

      // dot
      g.append('circle')
        .attr('cx', 0)
        .attr('cy', y)
        .attr('r', 6)
        .attr('fill', '#1e40af')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .transition()
        .duration(500)
        .delay(i * 80)
        .attr('cx', endX)

      // value label
      g.append('text')
        .attr('x', endX + 12)
        .attr('y', y)
        .attr('dy', '0.35em')
        .attr('fill', 'var(--text-primary)')
        .attr('font-size', '12px')
        .attr('font-weight', '600')
        .attr('font-variant-numeric', 'tabular-nums')
        .attr('opacity', 0)
        .text(d.correlation.toFixed(3))
        .transition()
        .duration(300)
        .delay(i * 80 + 400)
        .attr('opacity', 1)
    })

    // y-axis labels
    g.append('g')
      .call(d3.axisLeft(yScale).tickSize(0).tickFormat(d => FEATURE_LABELS[d] || d))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '13px')
      .attr('font-weight', '500')

    // x-axis
    g.append('g')
      .attr('transform', `translate(0,${chartHeight + 4})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d.toFixed(2)))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px')

    svg.append('text')
      .attr('x', margin.left + chartWidth / 2)
      .attr('y', containerHeight - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Correlation with Delay')
  }


  if (loading) {
    return (
      <section id="weather-impact" className="section section--alt">
        <div className="container">
          <p className="kicker">Weather Context</p>
          <h2>Weather Impact on Delays</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading weather data...
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section id="weather-impact" className="section section--alt">
        <div className="container">
          <p className="kicker">Weather Context</p>
          <h2>Weather Impact on Delays</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      </section>
    )
  }

  return (
    <section id="weather-impact" className="section section--alt">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Weather Context</p>
        <h2>Weather Impact on Delays</h2>
        <p style={{ marginBottom: 'var(--space-lg)' }}>
          Weather conditions correlate with arrival delays in our dataset. In our ablation study, adding weather features improved XGBoost forecast error by 10.3%, making it the most impactful feature group we tested. The data below aggregates weather conditions and arrival delays across 50 major U.S. domestic routes from January 2019 through June 2025.<sup><a href="#ref-3">3</a></sup><sup><a href="#ref-4">4</a></sup> We encoded 28 weather features, including hourly data aggregated into daily operating-hour metrics like peak wind, storm-hour counts, and departure/arrival-period conditions.
        </p>

        <div className="viz-card" style={{ height: 'auto', padding: 0 }}>
          <div style={{ padding: 'var(--space-md) var(--space-lg)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
            <div className="segmented-control">
              <button
                className={`segmented-control__btn${viewMode === 'impact' ? ' segmented-control__btn--active' : ''}`}
                onClick={() => setViewMode('impact')}
              >
                By Weather Type
              </button>
              <button
                className={`segmented-control__btn${viewMode === 'correlation' ? ' segmented-control__btn--active' : ''}`}
                onClick={() => setViewMode('correlation')}
              >
                Correlations
              </button>
            </div>
          </div>

          <div ref={chartWrapRef} style={{ padding: 'var(--space-lg)', background: 'var(--bg-base-elevated)' }}>
            {viewMode === 'impact' ? (
              <div ref={divergingRef} style={{ width: '100%', minHeight: '300px' }} />
            ) : (
              <div ref={lollipopRef} style={{ width: '100%', minHeight: '300px' }} />
            )}
          </div>
        </div>

        <div className="findings-grid findings-grid--4">
          <div className="finding-card">
            <h4>Snow Delays</h4>
            <p>Across {data?.by_severity?.find(d => d.label === 'Snow')?.n_flights?.toLocaleString()} flights during snow, the average delay is {data?.by_severity?.find(d => d.label === 'Snow')?.avg_delay?.toFixed(1)} minutes, nearly 20 minutes worse than clear conditions.</p>
          </div>
          <div className="finding-card">
            <h4>Clear Conditions</h4>
            <p>Routes during clear weather average {Math.abs(data?.by_severity?.find(d => d.label === 'Clear')?.avg_delay)?.toFixed(1)} minutes early, with the lowest proportion of high-delay days at {(data?.by_severity?.find(d => d.label === 'Clear')?.high_delay_pct * 100)?.toFixed(1)}%.</p>
          </div>
          <div className="finding-card">
            <h4>Strongest Correlation</h4>
            <p>Precipitation has the strongest correlation with delays ({data?.correlation?.precip?.toFixed(3)}), followed by severity index ({data?.correlation?.severity?.toFixed(3)}) and snowfall ({data?.correlation?.snowfall?.toFixed(3)}).</p>
          </div>
          <div className="finding-card">
            <h4>Ablation Testing</h4>
            <p>In our ablation test, adding 28 weather features improved XGBoost MAE by 10.3%, the largest gain from any feature group.</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default WeatherImpactChart
