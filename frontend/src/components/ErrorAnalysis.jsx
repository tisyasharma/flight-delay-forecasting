import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { sortModelEntries } from '../utils/helpers.js'
import { showTooltip, hideTooltip } from '../visualizations/tooltip.js'
import { METRIC_CONFIG } from '../utils/metricConfig.js'

function formatUnit(key) {
  const config = METRIC_CONFIG[key]
  if (!config || !config.unit) return ''
  return config.unit === '%' ? '%' : ` ${config.unit}`
}

function getQuarter(dateStr) {
  const month = new Date(dateStr).getMonth()
  if (month < 3) return 'Winter'
  if (month < 6) return 'Spring'
  if (month < 9) return 'Summer'
  return 'Fall'
}

const SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']

function ErrorAnalysis({ forecastData: rawData, loading, error }) {
  const routeChartRef = useRef(null)
  const dumbbellRef = useRef(null)
  const heatmapRef = useRef(null)
  const chartWrapRef = useRef(null)
  const [errorByRoute, setErrorByRoute] = useState([])
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [selectedMetric, setSelectedMetric] = useState('mae')
  const [viewMode, setViewMode] = useState('route')
  const [containerWidth, setContainerWidth] = useState(0)

  const availableModels = useMemo(() => {
    if (!rawData?.models) return []
    return sortModelEntries(Object.entries(rawData.models)).map(([key, model]) => ({
      key,
      name: model.name
    }))
  }, [rawData])

  const seasonalStats = useMemo(() => {
    if (!rawData?.predictions) return null

    const modelKey = selectedModel
    const predictions = rawData.predictions
    const routes = Object.keys(predictions)

    const seasonTotals = {}
    const seasonCounts = {}
    const routeSeasonMAE = {}

    routes.forEach(route => {
      const preds = predictions[route]
      if (!preds || preds.length === 0) return

      routeSeasonMAE[route] = {}
      SEASON_ORDER.forEach(season => {
        const seasonPreds = preds.filter(p => getQuarter(p.date) === season)
        if (seasonPreds.length === 0) return

        const errors = seasonPreds
          .filter(p => p[modelKey] != null && p.actual != null)
          .map(p => Math.abs(p[modelKey] - p.actual))

        if (errors.length > 0) {
          const mae = errors.reduce((a, b) => a + b, 0) / errors.length
          routeSeasonMAE[route][season] = mae

          if (!seasonTotals[season]) {
            seasonTotals[season] = 0
            seasonCounts[season] = 0
          }
          seasonTotals[season] += mae
          seasonCounts[season] += 1
        }
      })
    })

    const seasonAvgs = SEASON_ORDER.map(season => ({
      season,
      avg: seasonCounts[season] ? seasonTotals[season] / seasonCounts[season] : null
    })).filter(s => s.avg != null)

    if (seasonAvgs.length === 0) return null

    const bestSeason = seasonAvgs.reduce((a, b) => a.avg < b.avg ? a : b)
    const worstSeason = seasonAvgs.reduce((a, b) => a.avg > b.avg ? a : b)

    let worstRouteInWorstSeason = null
    let worstRouteMAE = 0
    routes.forEach(route => {
      const mae = routeSeasonMAE[route]?.[worstSeason.season]
      if (mae && mae > worstRouteMAE) {
        worstRouteMAE = mae
        worstRouteInWorstSeason = route
      }
    })

    return {
      bestSeason: bestSeason.season,
      bestSeasonMAE: bestSeason.avg,
      worstSeason: worstSeason.season,
      worstSeasonMAE: worstSeason.avg,
      worstRouteInWorstSeason,
      worstRouteMAE,
      spread: worstSeason.avg - bestSeason.avg
    }
  }, [rawData, selectedModel])

  const updateRouteData = (data, modelKey) => {
    const modelMetrics = data.models[modelKey]?.metrics?.by_route
    if (!modelMetrics) return

    const displayRoutes = new Set(data.routes || [])

    const routeData = Object.entries(modelMetrics)
      .filter(([route]) => displayRoutes.size === 0 || displayRoutes.has(route))
      .map(([route, metrics]) => ({
        route,
        mae: metrics.mae,
        within_15: metrics.within_15,
        rmse: metrics.rmse,
        mape: metrics.mape,
        median_ae: metrics.median_ae,
        threshold_acc: metrics.threshold_acc,
        r2: metrics.r2,
      }))

    setErrorByRoute(routeData)
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
  }, [loading])

  useEffect(() => {
    if (!rawData) return
    updateRouteData(rawData, selectedModel)
  }, [selectedModel, rawData])

  useEffect(() => {
    if (routeChartRef.current && errorByRoute.length > 0 && viewMode === 'route') {
      renderRouteChart()
    }
  }, [errorByRoute, selectedMetric, viewMode, containerWidth])

  useEffect(() => {
    if (rawData && dumbbellRef.current && viewMode === 'dumbbell') {
      renderDumbbellChart()
    }
  }, [rawData, viewMode, containerWidth])

  useEffect(() => {
    if (rawData && heatmapRef.current && viewMode === 'heatmap') {
      renderHeatmap()
    }
  }, [rawData, viewMode, containerWidth, selectedModel])

  const renderRouteChart = () => {
    const container = routeChartRef.current
    container.innerHTML = ''

    const margin = { top: 20, right: 30, bottom: 80, left: 50 }
    const width = container.clientWidth - margin.left - margin.right
    const height = 300 - margin.top - margin.bottom

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const metric = selectedMetric
    const sortedData = [...errorByRoute].sort((a, b) => a[metric] - b[metric])

    const x = d3.scaleBand()
      .domain(sortedData.map(d => d.route))
      .range([0, width])
      .padding(0.25)

    const y = d3.scaleLinear()
      .domain([0, d3.max(sortedData, d => d[metric]) * 1.15])
      .range([height, 0])

    const higherIsBetter = ['within_15', 'threshold_acc', 'r2'].includes(metric)
    const colorScale = d3.scaleLinear()
      .domain([d3.min(sortedData, d => d[metric]), d3.max(sortedData, d => d[metric])])
      .range(higherIsBetter ? ['#c8102e', '#1e40af'] : ['#1e40af', '#c8102e'])

    svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(5).tickSize(-width).tickFormat(''))
      .selectAll('line')
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-dasharray', '2,2')

    svg.selectAll('.grid .domain').remove()

    const bars = svg.selectAll('rect')
      .data(sortedData)
      .enter()
      .append('rect')
      .attr('x', d => x(d.route))
      .attr('y', height)
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('fill', d => colorScale(d[metric]))
      .attr('opacity', 0.85)
      .attr('rx', 3)
      .style('cursor', 'pointer')

    bars.transition()
      .duration(400)
      .delay((d, i) => i * 40)
      .attr('y', d => y(d[metric]))
      .attr('height', d => height - y(d[metric]))

    bars.on('mouseenter', function(event, d) {
        d3.select(this).attr('opacity', 1)
        const html = `
          <div><strong>${d.route}</strong></div>
          <div>MAE: ${d.mae?.toFixed(1) ?? '-'} min</div>
          <div>Hit Rate: ${d.within_15?.toFixed(1) ?? '-'}%</div>
          <div>RMSE: ${d.rmse?.toFixed(1) ?? '-'} min</div>
          <div>Delay Detection: ${d.threshold_acc?.toFixed(1) ?? '-'}%</div>
          <div>R\u00B2: ${d.r2?.toFixed(3) ?? '-'}</div>
        `
        showTooltip(event, html)
      })
      .on('mousemove', function(event, d) {
        const html = `
          <div><strong>${d.route}</strong></div>
          <div>MAE: ${d.mae?.toFixed(1) ?? '-'} min</div>
          <div>Hit Rate: ${d.within_15?.toFixed(1) ?? '-'}%</div>
          <div>RMSE: ${d.rmse?.toFixed(1) ?? '-'} min</div>
          <div>Delay Detection: ${d.threshold_acc?.toFixed(1) ?? '-'}%</div>
          <div>R\u00B2: ${d.r2?.toFixed(3) ?? '-'}</div>
        `
        showTooltip(event, html)
      })
      .on('mouseleave', function() {
        d3.select(this).attr('opacity', 0.85)
        hideTooltip()
      })

    svg.selectAll('.label')
      .data(sortedData)
      .enter()
      .append('text')
      .attr('x', d => x(d.route) + x.bandwidth() / 2)
      .attr('y', d => y(d[metric]) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '9px')
      .attr('font-weight', '600')
      .attr('opacity', 0)
      .text(d => d[metric] != null ? d[metric].toFixed(1) : '')
      .transition()
      .duration(300)
      .delay((d, i) => i * 40 + 300)
      .attr('opacity', 1)

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .style('fill', 'var(--text-primary)')
      .style('font-size', '9px')
      .style('font-weight', '500')
      .attr('transform', 'rotate(-40)')
      .attr('text-anchor', 'end')
      .attr('dx', '-0.5em')
      .attr('dy', '0.3em')

    const formatTick = (d) => {
      if (metric === 'r2') return d.toFixed(2)
      if (metric === 'within_15' || metric === 'threshold_acc') return d.toFixed(0) + '%'
      return d.toFixed(0)
    }
    svg.append('g')
      .call(d3.axisLeft(y).ticks(5).tickFormat(formatTick))
      .selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '11px')

    svg.selectAll('.domain')
      .attr('stroke', 'var(--border)')
  }

  /* Dumbbell chart: gradient boosting MAE vs deep learning MAE per route */
  const renderDumbbellChart = () => {
    const container = d3.select(dumbbellRef.current)
    container.selectAll('*').remove()

    if (!rawData?.models) return

    const gbModels = ['xgboost', 'lightgbm'].filter(k => rawData.models[k])
    const dlModels = ['lstm', 'tcn'].filter(k => rawData.models[k])

    if (gbModels.length === 0 || dlModels.length === 0) return

    const routes = rawData.routes || Object.keys(rawData.predictions || {})
    const dumbbellData = routes.map(route => {
      const gbMAEs = gbModels.map(k => rawData.models[k].metrics.by_route[route]?.mae).filter(v => v != null)
      const dlMAEs = dlModels.map(k => rawData.models[k].metrics.by_route[route]?.mae).filter(v => v != null)

      const gbMean = gbMAEs.length > 0 ? gbMAEs.reduce((a, b) => a + b, 0) / gbMAEs.length : null
      const dlMean = dlMAEs.length > 0 ? dlMAEs.reduce((a, b) => a + b, 0) / dlMAEs.length : null

      return { route, gb: gbMean, dl: dlMean }
    }).filter(d => d.gb != null && d.dl != null)
      .sort((a, b) => a.gb - b.gb)

    const containerWidth = dumbbellRef.current.clientWidth
    const rowHeight = 21
    const containerHeight = dumbbellData.length * rowHeight + 74

    const margin = { top: 32, right: 50, bottom: 36, left: 100 }
    const chartWidth = containerWidth - margin.left - margin.right
    const chartHeight = dumbbellData.length * rowHeight

    const svg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    const maxVal = d3.max(dumbbellData, d => Math.max(d.gb, d.dl)) * 1.12
    const xScale = d3.scaleLinear()
      .domain([0, maxVal])
      .range([0, chartWidth])

    const yScale = d3.scaleBand()
      .domain(dumbbellData.map(d => d.route))
      .range([0, chartHeight])
      .padding(0.2)

    const ticks = xScale.ticks(6)
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

    dumbbellData.forEach((d, i) => {
      const y = yScale(d.route) + yScale.bandwidth() / 2
      const x1 = xScale(Math.min(d.gb, d.dl))
      const x2 = xScale(Math.max(d.gb, d.dl))

      g.append('line')
        .attr('x1', x1)
        .attr('y1', y)
        .attr('x2', x1)
        .attr('y2', y)
        .attr('stroke', '#9ca3af')
        .attr('stroke-width', 2)
        .attr('opacity', 0.5)
        .transition()
        .duration(400)
        .delay(i * 50)
        .attr('x2', x2)

      g.append('circle')
        .attr('cx', 0)
        .attr('cy', y)
        .attr('r', 4)
        .attr('fill', '#1e40af')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseenter', function(event) {
          showTooltip(event, `<div><strong>${d.route}</strong></div><div>Gradient Boosting MAE: ${d.gb.toFixed(1)} min</div>`)
        })
        .on('mouseleave', hideTooltip)
        .transition()
        .duration(400)
        .delay(i * 50)
        .attr('cx', xScale(d.gb))

      g.append('circle')
        .attr('cx', 0)
        .attr('cy', y)
        .attr('r', 4)
        .attr('fill', '#c8102e')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseenter', function(event) {
          showTooltip(event, `<div><strong>${d.route}</strong></div><div>Deep Learning MAE: ${d.dl.toFixed(1)} min</div>`)
        })
        .on('mouseleave', hideTooltip)
        .transition()
        .duration(400)
        .delay(i * 50)
        .attr('cx', xScale(d.dl))
    })

    g.append('g')
      .call(d3.axisLeft(yScale).tickSize(0))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '10px')
      .attr('font-weight', '500')

    g.append('g')
      .attr('transform', `translate(0,${chartHeight + 4})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d => `${d.toFixed(0)}`))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px')

    svg.append('text')
      .attr('x', margin.left + chartWidth / 2)
      .attr('y', containerHeight - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px')
      .text('MAE (minutes)')

    const legendX = margin.left + chartWidth - 180
    const legendY = 10

    svg.append('circle').attr('cx', legendX).attr('cy', legendY).attr('r', 4).attr('fill', '#1e40af')
    svg.append('text').attr('x', legendX + 10).attr('y', legendY + 4).text('Gradient Boosting (XGBoost + LightGBM)')
      .attr('fill', 'var(--text-muted)').attr('font-size', '10px')

    svg.append('circle').attr('cx', legendX).attr('cy', legendY + 14).attr('r', 4).attr('fill', '#c8102e')
    svg.append('text').attr('x', legendX + 10).attr('y', legendY + 18).text('Deep Learning (LSTM + TCN)')
      .attr('fill', 'var(--text-muted)').attr('font-size', '10px')
  }

  const renderHeatmap = () => {
    const container = d3.select(heatmapRef.current)
    container.selectAll('*').remove()

    if (!rawData?.predictions) return

    const modelKey = selectedModel
    const predictions = rawData.predictions
    const routes = Object.keys(predictions)

    const seasonalMAE = {}
    routes.forEach(route => {
      const preds = predictions[route]
      if (!preds || preds.length === 0) return

      seasonalMAE[route] = {}
      SEASON_ORDER.forEach(season => {
        const seasonPreds = preds.filter(p => getQuarter(p.date) === season)
        if (seasonPreds.length === 0) return

        const errors = seasonPreds
          .filter(p => p[modelKey] != null && p.actual != null)
          .map(p => Math.abs(p[modelKey] - p.actual))

        if (errors.length > 0) {
          seasonalMAE[route][season] = errors.reduce((a, b) => a + b, 0) / errors.length
        }
      })
    })

    const routeOverallMAE = routes.map(route => {
      const allErrors = Object.values(seasonalMAE[route] || {})
      const avg = allErrors.length > 0 ? allErrors.reduce((a, b) => a + b, 0) / allErrors.length : 999
      return { route, avg }
    }).sort((a, b) => a.avg - b.avg)

    const sortedRoutes = routeOverallMAE.map(d => d.route)

    const allValues = []
    sortedRoutes.forEach(route => {
      SEASON_ORDER.forEach(season => {
        const val = seasonalMAE[route]?.[season]
        if (val != null) allValues.push(val)
      })
    })

    if (allValues.length === 0) return

    const containerWidth = heatmapRef.current.clientWidth
    const cellHeight = 23
    const containerHeight = sortedRoutes.length * cellHeight + 74

    const margin = { top: 32, right: 40, bottom: 24, left: 100 }
    const chartWidth = containerWidth - margin.left - margin.right
    const chartHeight = sortedRoutes.length * cellHeight

    const svg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`)

    const xScale = d3.scaleBand()
      .domain(SEASON_ORDER)
      .range([0, chartWidth])
      .padding(0.06)

    const yScale = d3.scaleBand()
      .domain(sortedRoutes)
      .range([0, chartHeight])
      .padding(0.06)

    const colorScale = d3.scaleSequential()
      .domain([d3.min(allValues), d3.max(allValues)])
      .interpolator(d3.interpolateRgbBasis(['#10b981', '#f59e0b', '#ef4444']))

    sortedRoutes.forEach((route, ri) => {
      SEASON_ORDER.forEach((season, si) => {
        const val = seasonalMAE[route]?.[season]
        if (val == null) return

        g.append('rect')
          .attr('x', xScale(season))
          .attr('y', yScale(route))
          .attr('width', xScale.bandwidth())
          .attr('height', yScale.bandwidth())
          .attr('fill', colorScale(val))
          .attr('rx', 3)
          .attr('opacity', 0)
          .style('cursor', 'pointer')
          .on('mouseenter', function(event) {
            d3.select(this).attr('stroke', 'var(--text-primary)').attr('stroke-width', 1.5)
            showTooltip(event, `<div><strong>${route}</strong></div><div>${season}: ${val.toFixed(1)} min MAE</div>`)
          })
          .on('mouseleave', function() {
            d3.select(this).attr('stroke', 'none')
            hideTooltip()
          })
          .transition()
          .duration(300)
          .delay(ri * 30 + si * 60)
          .attr('opacity', 0.9)

        g.append('text')
          .attr('x', xScale(season) + xScale.bandwidth() / 2)
          .attr('y', yScale(route) + yScale.bandwidth() / 2)
          .attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', '#1e293b')
          .attr('font-size', '10px')
          .attr('font-weight', '600')
          .attr('font-variant-numeric', 'tabular-nums')
          .attr('pointer-events', 'none')
          .attr('opacity', 0)
          .text(val.toFixed(1))
          .transition()
          .duration(300)
          .delay(ri * 30 + si * 60 + 150)
          .attr('opacity', 1)
      })
    })

    g.append('g')
      .call(d3.axisLeft(yScale).tickSize(0))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '10px')
      .attr('font-weight', '500')

    g.append('g')
      .attr('transform', `translate(0, -6)`)
      .call(d3.axisTop(xScale).tickSize(0))
      .call(g => g.select('.domain').remove())
      .selectAll('text')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '11px')
      .attr('font-weight', '600')

    const legendWidth = 160
    const legendHeight = 10
    const legendX = containerWidth - margin.right - legendWidth
    const legendY = containerHeight - 18

    const defs = svg.append('defs')
    const linearGradient = defs.append('linearGradient').attr('id', 'heatmap-gradient')
    linearGradient.append('stop').attr('offset', '0%').attr('stop-color', '#10b981')
    linearGradient.append('stop').attr('offset', '50%').attr('stop-color', '#f59e0b')
    linearGradient.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444')

    svg.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('rx', 3)
      .style('fill', 'url(#heatmap-gradient)')

    svg.append('text')
      .attr('x', legendX)
      .attr('y', legendY - 4)
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '9px')
      .text(`${d3.min(allValues).toFixed(0)} min`)

    svg.append('text')
      .attr('x', legendX + legendWidth)
      .attr('y', legendY - 4)
      .attr('text-anchor', 'end')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '9px')
      .text(`${d3.max(allValues).toFixed(0)} min`)

    svg.append('text')
      .attr('x', legendX + legendWidth / 2)
      .attr('y', legendY - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '9px')
      .text('MAE')
  }

  if (loading) {
    return (
      <section id="error-analysis" className="section">
        <div className="container">
          <p className="kicker">Analysis</p>
          <h2>Error Analysis</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading error analysis data...
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section id="error-analysis" className="section">
        <div className="container">
          <p className="kicker">Analysis</p>
          <h2>Error Analysis</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      </section>
    )
  }

  const metric = selectedMetric
  const config = METRIC_CONFIG[metric]
  const higherIsBetter = ['within_15', 'threshold_acc', 'r2'].includes(metric)
  const sortedRoutes = [...errorByRoute].sort((a, b) =>
    higherIsBetter ? b[metric] - a[metric] : a[metric] - b[metric]
  )
  const bestRoute = sortedRoutes[0]
  const worstRoute = sortedRoutes[sortedRoutes.length - 1]
  const avgValue = errorByRoute.length > 0
    ? (errorByRoute.reduce((sum, r) => sum + (r[metric] || 0), 0) / errorByRoute.length).toFixed(1)
    : 0

  const formatValue = (val) => {
    if (val == null) return '-'
    if (metric === 'r2') return val.toFixed(3)
    return val.toFixed(1)
  }

  const selectedModelName = availableModels.find(m => m.key === selectedModel)?.name || selectedModel

  const viewDescriptions = {
    route: `Forecast accuracy varies across routes. ${higherIsBetter
      ? `Higher ${config.label} indicates stronger performance.`
      : `Lower ${config.label} indicates more accurate forecasts.`}`,
    dumbbell: 'Gradient boosting outperforms deep learning on every route in the test set, though the margin varies by corridor.',
    heatmap: `${selectedModelName} forecast error (MAE) by season and route. Seasonal patterns vary by model and corridor.`
  }

  return (
    <section id="error-analysis" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Analysis</p>
        <h2>Error Analysis</h2>
        <p style={{ marginBottom: 'var(--space-lg)' }}>
          Not all routes are equally predictable. Forecast errors vary widely across the top 20 routes by traffic volume during the test period (July 2024 through June 2025). {viewDescriptions[viewMode]}
        </p>

        <div className="viz-card" style={{ height: 'auto', padding: 0 }}>
          <div style={{ display: 'flex', padding: 'var(--space-md) var(--space-lg)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)', flexWrap: 'wrap', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <div className="segmented-control">
              <button
                className={`segmented-control__btn${viewMode === 'route' ? ' segmented-control__btn--active' : ''}`}
                onClick={() => setViewMode('route')}
              >
                By Route
              </button>
              <button
                className={`segmented-control__btn${viewMode === 'dumbbell' ? ' segmented-control__btn--active' : ''}`}
                onClick={() => setViewMode('dumbbell')}
              >
                Gradient Boosting vs Deep Learning
              </button>
              <button
                className={`segmented-control__btn${viewMode === 'heatmap' ? ' segmented-control__btn--active' : ''}`}
                onClick={() => setViewMode('heatmap')}
              >
                Seasonal Patterns
              </button>
            </div>

            {viewMode === 'route' && (
              <div style={{ display: 'flex', gap: 'var(--space-sm)', marginLeft: 'auto', alignItems: 'center' }}>
                <select
                  className="select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {availableModels.map(model => (
                    <option key={model.key} value={model.key}>{model.name}</option>
                  ))}
                </select>
                <select
                  className="select"
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  <option value="mae">MAE</option>
                  <option value="within_15">Hit Rate</option>
                  <option value="rmse">RMSE</option>
                </select>
              </div>
            )}

            {viewMode === 'heatmap' && (
              <div style={{ display: 'flex', gap: 'var(--space-sm)', marginLeft: 'auto', alignItems: 'center' }}>
                <select
                  className="select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {availableModels.map(model => (
                    <option key={model.key} value={model.key}>{model.name}</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div ref={chartWrapRef} style={{ padding: 'var(--space-lg)', background: 'var(--bg-base-elevated)' }}>
            {viewMode === 'route' && (
              <div ref={routeChartRef} style={{ width: '100%', height: '300px' }} />
            )}
            {viewMode === 'dumbbell' && (
              <div ref={dumbbellRef} style={{ width: '100%', minHeight: '300px' }} />
            )}
            {viewMode === 'heatmap' && (
              <div ref={heatmapRef} style={{ width: '100%', minHeight: '300px' }} />
            )}
          </div>
        </div>

        {viewMode === 'route' && (
          <div className="findings-grid findings-grid--3">
            <div className="finding-card finding-card--green">
              <h4>Hawaii Routes Lead</h4>
              <p>
                Inter-island routes like HNL-OGG consistently show the lowest forecast errors. Stable tropical weather patterns and shorter flight distances make these corridors far more predictable than mainland routes.
              </p>
            </div>
            <div className="finding-card finding-card--cyan">
              <h4>Wide Variation</h4>
              <p>
                {(() => {
                  const maeValues = errorByRoute.map(r => r.mae).filter(v => v != null)
                  const minMAE = Math.min(...maeValues)
                  const maxMAE = Math.max(...maeValues)
                  const ratio = (maxMAE / minMAE).toFixed(1)
                  return `Forecast difficulty varies by up to ${ratio}x across routes. The easiest routes have MAE under 5 minutes, while the hardest exceed 15 minutes.`
                })()}
              </p>
            </div>
            <div className="finding-card finding-card--red">
              <h4>Atlanta Hub and Northeast Struggle</h4>
              <p>
                Routes involving Atlanta (FLL-ATL, MCO-ATL) and the Northeast corridor (DCA-BOS, LGA-ORD) show the highest errors, exceeding 14 minutes MAE. Weather volatility and hub congestion make these corridors harder to forecast.
              </p>
            </div>
          </div>
        )}

        {viewMode === 'dumbbell' && (
          <div className="findings-grid findings-grid--3">
            <div className="finding-card finding-card--blue">
              <h4>Consistent Advantage</h4>
              <p>Gradient boosting outperforms deep learning on every route in the test set, with an average margin of 2.8 minutes MAE. The advantage holds regardless of route characteristics.</p>
            </div>
            <div className="finding-card finding-card--cyan">
              <h4>Largest Gaps</h4>
              <p>East Coast routes show the biggest performance difference. ORD-LGA and DCA-BOS see gaps of nearly 3 minutes, where weather volatility may favor gradient boosting's feature engineering.</p>
            </div>
            <div className="finding-card finding-card--green">
              <h4>Smallest Gaps</h4>
              <p>Hawaii routes (OGG-HNL, HNL-OGG) show the smallest gaps at under 0.5 minutes. With stable weather and predictable delays, both model types perform similarly well.</p>
            </div>
          </div>
        )}

        {viewMode === 'heatmap' && seasonalStats && (
          <div className="findings-grid findings-grid--3">
            <div className="finding-card finding-card--green">
              <h4>Easiest Season: {seasonalStats.bestSeason}</h4>
              <p>{selectedModelName} performs best in {seasonalStats.bestSeason.toLowerCase()} with an average error of {seasonalStats.bestSeasonMAE.toFixed(1)} minutes. This may reflect more predictable conditions during this period.</p>
            </div>
            <div className="finding-card finding-card--cyan">
              <h4>Seasonal Spread</h4>
              <p>The gap between best and worst seasons is {seasonalStats.spread.toFixed(1)} minutes. {seasonalStats.spread > 5 ? 'This variation may reflect differing weather and travel patterns across seasons.' : 'Relatively consistent performance across seasons.'}</p>
            </div>
            <div className="finding-card finding-card--red">
              <h4>Hardest Season: {seasonalStats.worstSeason}</h4>
              <p>{seasonalStats.worstSeason} is most challenging with {seasonalStats.worstSeasonMAE.toFixed(1)} min average error. {seasonalStats.worstRouteInWorstSeason} is the toughest route this season at {seasonalStats.worstRouteMAE.toFixed(1)} min.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default ErrorAnalysis
