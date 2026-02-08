import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { sortModelEntries } from '../utils/helpers.js'
import { showTooltip, hideTooltip } from '../visualizations/tooltip.js'
import { extractCityPairs, buildDirectionalRoute, formatCityPairDisplay } from '../utils/routeUtils.js'
import DirectionToggle from './DirectionToggle.jsx'

const parseDate = d3.timeParse('%Y-%m-%d')
const formatAxisDate = d3.timeFormat('%b %Y')
const formatWindowDate = d3.timeFormat('%b %d, %Y')
const formatTooltipDate = d3.timeFormat('%a, %b %d, %Y')

const MODEL_DETAILS = {
  xgboost: {
    architecture: 'Gradient Boosting, Optuna-tuned',
    type: 'Gradient Boosting',
    params: '200 trees, max depth 10, lr 0.029, subsample 0.63',
    training: 'Bayesian-optimized (50 Optuna trials), early stopping',
    features: '63 features: temporal, lag (1-28d), rolling stats, route, and 28 weather features'
  },
  lightgbm: {
    architecture: 'Leaf-wise Gradient Boosting, Optuna-tuned',
    type: 'Gradient Boosting',
    params: '550 trees, 123 leaves, lr 0.022, subsample 0.76',
    training: 'Bayesian-optimized (50 Optuna trials), early stopping',
    features: '63 features: temporal, lag (1-28d), rolling stats, route, and 28 weather features'
  },
  lstm: {
    architecture: 'LSTM with temporal attention, Optuna-tuned',
    type: 'Deep Learning',
    params: 'Hidden 96, 1 layer, dropout 0.14, target-scaled',
    training: 'AdamW (lr 0.00023, wd 5e-5), 50 Optuna trials, epoch-level pruning',
    features: '26 features: calendar, weather conditions, and target history in 28-day sequences'
  },
  tcn: {
    architecture: 'Dilated causal convolutions, Optuna-tuned',
    type: 'Deep Learning',
    params: 'Depth 4 [64,64,128,128], kernel 4, dropout 0.36, target-scaled',
    training: 'AdamW (lr 0.0024, wd 1e-4), 50 Optuna trials, epoch-level pruning',
    features: '26 features: calendar, weather conditions, and target history in 28-day sequences'
  },
  naive: {
    architecture: 'Previous day delay (lag-1)',
    type: 'Baseline',
    params: null,
    training: null,
    features: 'Single feature: yesterday\'s arrival delay'
  },
  ma: {
    architecture: '7-day rolling mean',
    type: 'Baseline',
    params: null,
    training: null,
    features: 'Single feature: 7-day rolling average of arrival delay'
  }
}

function ForecastChart({ forecastData, loading, error }) {
  const mainChartRef = useRef(null)
  const brushChartRef = useRef(null)
  const chartAreaRef = useRef(null)
  const [selectedCityPair, setSelectedCityPair] = useState(null)
  const [direction, setDirection] = useState('outbound')
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [brushExtent, setBrushExtent] = useState(null)
  const [showMethodology, setShowMethodology] = useState(false)
  const [containerWidth, setContainerWidth] = useState(0)
  const brushRef = useRef(null)
  const chartRef = useRef(null)

  const cityPairs = useMemo(() => {
    if (!forecastData?.routes) return []
    const pairs = extractCityPairs(forecastData.routes)

    // sort by best model hit rate (avg of both directions), highest first
    const bestModelKey = 'xgboost'
    const byRoute = forecastData.models?.[bestModelKey]?.metrics?.by_route
    if (!byRoute) return pairs

    return pairs.sort((a, b) => {
      const avgHitRate = (pair) => {
        const [c1, c2] = pair.split('-')
        const r1 = byRoute[`${c1}-${c2}`]?.within_15
        const r2 = byRoute[`${c2}-${c1}`]?.within_15
        const vals = [r1, r2].filter(v => v != null)
        return vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
      }
      return avgHitRate(b) - avgHitRate(a)
    })
  }, [forecastData])

  const selectedRoute = useMemo(() => {
    if (!selectedCityPair) return null
    return buildDirectionalRoute(selectedCityPair, direction)
  }, [selectedCityPair, direction])

  useEffect(() => {
    const el = chartAreaRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = Math.round(entries[0].contentRect.width)
      setContainerWidth(prev => prev !== w ? w : prev)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [loading])

  useEffect(() => {
    if (forecastData?.routes?.length && !selectedCityPair) {
      const pairs = extractCityPairs(forecastData.routes)
      if (pairs.length > 0) {
        setSelectedCityPair(pairs[0])
      }
    }
  }, [forecastData])

  const fullData = useMemo(() => {
    if (!forecastData || !selectedRoute) return null

    const historical = forecastData.historical?.[selectedRoute] || []
    const predictions = forecastData.predictions?.[selectedRoute] || []

    const historicalParsed = historical.map(h => ({
      date: parseDate(h.date),
      dateStr: h.date,
      actual: h.actual,
      predicted: null,
      isTest: false
    })).filter(d => d.date)

    const testParsed = predictions.map(p => ({
      date: parseDate(p.date),
      dateStr: p.date,
      actual: p.actual,
      predicted: p[selectedModel] ?? null,
      isTest: true
    })).filter(d => d.date)

    return [...historicalParsed, ...testParsed].sort((a, b) => a.date - b.date)
  }, [forecastData, selectedRoute, selectedModel])

  const testStartDate = useMemo(() => {
    if (!forecastData?.test_period?.start) return null
    return parseDate(forecastData.test_period.start)
  }, [forecastData])

  const metrics = useMemo(() => {
    if (!forecastData || !selectedRoute) return null
    const modelMetrics = forecastData.models[selectedModel]?.metrics
    if (!modelMetrics) return null
    return {
      overall: modelMetrics.overall,
      route: modelMetrics.by_route?.[selectedRoute]
    }
  }, [forecastData, selectedRoute, selectedModel])

  const dateExtent = useMemo(() => {
    if (!fullData?.length) return null
    return d3.extent(fullData, d => d.date)
  }, [fullData])

  const defaultExtent = useMemo(() => {
    if (!testStartDate || !dateExtent) return dateExtent
    const sixMonthsMs = 180 * 24 * 60 * 60 * 1000
    const endDate = new Date(Math.min(
      testStartDate.getTime() + sixMonthsMs,
      dateExtent[1].getTime()
    ))
    return [testStartDate, endDate]
  }, [testStartDate, dateExtent])

  const handleReset = () => {
    setBrushExtent(defaultExtent)
  }

  const handleShowAll = () => {
    setBrushExtent(dateExtent)
  }

  const handleZoomIn = () => {
    const extent = brushExtent || defaultExtent
    if (!extent || !dateExtent) return
    const [start, end] = extent
    const range = end - start
    const center = new Date((start.getTime() + end.getTime()) / 2)
    const newRange = range * 0.5
    const newStart = new Date(center.getTime() - newRange / 2)
    const newEnd = new Date(center.getTime() + newRange / 2)
    setBrushExtent([
      new Date(Math.max(newStart.getTime(), dateExtent[0].getTime())),
      new Date(Math.min(newEnd.getTime(), dateExtent[1].getTime()))
    ])
  }

  const handleZoomOut = () => {
    const extent = brushExtent || defaultExtent
    if (!extent || !dateExtent) return
    const [start, end] = extent
    const range = end - start
    const center = new Date((start.getTime() + end.getTime()) / 2)
    const newRange = range * 2
    const newStart = new Date(center.getTime() - newRange / 2)
    const newEnd = new Date(center.getTime() + newRange / 2)
    setBrushExtent([
      new Date(Math.max(newStart.getTime(), dateExtent[0].getTime())),
      new Date(Math.min(newEnd.getTime(), dateExtent[1].getTime()))
    ])
  }

  useEffect(() => {
    if (!fullData?.length || !dateExtent || !mainChartRef.current || !brushChartRef.current) return

    const mainContainer = mainChartRef.current
    const brushContainer = brushChartRef.current
    mainContainer.innerHTML = ''
    brushContainer.innerHTML = ''

    const margin = { top: 20, right: 30, bottom: 30, left: 55 }
    const brushMargin = { top: 5, right: 30, bottom: 20, left: 55 }

    const mainWidth = mainContainer.clientWidth - margin.left - margin.right
    const mainHeight = 320 - margin.top - margin.bottom
    const brushHeight = 60 - brushMargin.top - brushMargin.bottom

    if (mainWidth <= 0) return

    const xScaleFull = d3.scaleTime()
      .domain(dateExtent)
      .range([0, mainWidth])

    const currentExtent = brushExtent || defaultExtent || dateExtent

    const xScaleMain = d3.scaleTime()
      .domain(currentExtent)
      .range([0, mainWidth])

    const visibleData = fullData.filter(d => d.date >= currentExtent[0] && d.date <= currentExtent[1])
    const yMin = d3.min(visibleData, d => Math.min(d.actual ?? 0, d.predicted ?? d.actual ?? 0)) || 0
    const yMax = d3.max(visibleData, d => Math.max(d.actual || 0, d.predicted || 0)) || 50
    const yScale = d3.scaleLinear()
      .domain([Math.min(0, yMin * 1.15), yMax * 1.15])
      .range([mainHeight, 0])

    const mainSvg = d3.select(mainContainer)
      .append('svg')
      .attr('width', mainWidth + margin.left + margin.right)
      .attr('height', mainHeight + margin.top + margin.bottom)

    const mainG = mainSvg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    mainSvg.append('defs').append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', mainWidth)
      .attr('height', mainHeight)

    mainG.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).ticks(6).tickSize(-mainWidth).tickFormat(''))
      .selectAll('line')
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-dasharray', '2,2')
    mainG.selectAll('.grid .domain').remove()

    const chartArea = mainG.append('g').attr('clip-path', 'url(#clip)')

    const splitX = testStartDate ? xScaleMain(testStartDate) : 0
    const showSplit = testStartDate && testStartDate >= currentExtent[0] && testStartDate <= currentExtent[1]

    chartArea.append('rect')
      .attr('class', 'train-region')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', showSplit ? splitX : 0)
      .attr('height', mainHeight)
      .attr('fill', 'var(--bg-base-soft)')
      .attr('opacity', 0.3)
      .style('display', showSplit ? null : 'none')

    chartArea.append('line')
      .attr('class', 'split-line')
      .attr('x1', splitX)
      .attr('x2', splitX)
      .attr('y1', 0)
      .attr('y2', mainHeight)
      .attr('stroke', 'var(--red)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .style('display', showSplit ? null : 'none')

    const lineActual = d3.line()
      .defined(d => d.actual != null)
      .x(d => xScaleMain(d.date))
      .y(d => yScale(d.actual))
      .curve(d3.curveMonotoneX)

    const linePred = d3.line()
      .defined(d => d.predicted != null)
      .x(d => xScaleMain(d.date))
      .y(d => yScale(d.predicted))
      .curve(d3.curveMonotoneX)

    const testData = visibleData.filter(d => d.isTest && d.predicted != null)

    if (yMin < 0) {
      chartArea.append('line')
        .attr('class', 'zero-line')
        .attr('x1', 0)
        .attr('x2', mainWidth)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', 'var(--text-muted)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4')
        .attr('opacity', 0.5)
    }

    chartArea.append('path')
      .attr('class', 'line-actual')
      .datum(visibleData)
      .attr('fill', 'none')
      .attr('stroke', 'var(--green)')
      .attr('stroke-width', 1.5)
      .attr('d', lineActual)

    chartArea.append('path')
      .attr('class', 'line-pred')
      .datum(testData)
      .attr('fill', 'none')
      .attr('stroke', 'var(--blue)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,2')
      .attr('d', linePred)

    const bisect = d3.bisector(d => d.date).center
    const focusLine = mainG.append('line')
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-dasharray', '3,3')
      .attr('y1', 0)
      .attr('y2', mainHeight)
      .style('display', 'none')

    const focusActual = mainG.append('circle').attr('r', 4).attr('fill', 'var(--green)').attr('stroke', '#fff').attr('stroke-width', 1.5).style('display', 'none')
    const focusPred = mainG.append('circle').attr('r', 4).attr('fill', 'var(--blue)').attr('stroke', '#fff').attr('stroke-width', 1.5).style('display', 'none')

    mainG.append('rect')
      .attr('fill', 'transparent')
      .attr('width', mainWidth)
      .attr('height', mainHeight)
      .style('cursor', 'crosshair')
      .on('mousemove', (event) => {
        if (!chartRef.current) return
        const { xScaleMain: scale, yScale: yS, visibleData: vd, focusLine: fLine, focusActual: fActual, focusPred: fPred, bisect: bis, mainContainer: container } = chartRef.current

        const [mx] = d3.pointer(event)
        const date = scale.invert(mx)
        const i = bis(vd, date)
        const d = vd[Math.min(i, vd.length - 1)]
        if (!d) return

        fLine.style('display', null).attr('x1', scale(d.date)).attr('x2', scale(d.date))

        if (d.actual != null) {
          fActual.style('display', null).attr('cx', scale(d.date)).attr('cy', yS(d.actual))
        } else {
          fActual.style('display', 'none')
        }

        if (d.predicted != null) {
          fPred.style('display', null).attr('cx', scale(d.date)).attr('cy', yS(d.predicted))
        } else {
          fPred.style('display', 'none')
        }

        let html = `<div><strong>${formatTooltipDate(d.date)}</strong></div>`
        if (d.actual != null) {
          html += `<div style="color: var(--green);">${d.actual.toFixed(1)} min actual</div>`
        }
        if (d.predicted != null) {
          html += `<div style="color: var(--blue);">${d.predicted.toFixed(1)} min forecast</div>`
          if (d.actual != null) {
            const err = d.predicted - d.actual
            html += `<div>Error: ${err >= 0 ? '+' : ''}${err.toFixed(1)} min</div>`
          }
        }
        html += `<div style="color: var(--text-muted); font-size: 11px;">${d.isTest ? 'Test Period' : 'Training Period'}</div>`
        showTooltip(event, html, { bounds: container })
      })
      .on('mouseleave', () => {
        if (!chartRef.current) return
        const { focusLine: fLine, focusActual: fActual, focusPred: fPred } = chartRef.current
        fLine.style('display', 'none')
        fActual.style('display', 'none')
        fPred.style('display', 'none')
        hideTooltip()
      })

    const xAxisG = mainG.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${mainHeight})`)
      .call(d3.axisBottom(xScaleMain).ticks(6).tickFormat(formatAxisDate))
    xAxisG.selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '11px')

    chartRef.current = {
      xScaleMain,
      yScale,
      chartArea,
      xAxisG,
      lineActual,
      linePred,

      mainHeight,
      mainWidth,
      testStartDate,
      visibleData,
      focusLine,
      focusActual,
      focusPred,
      bisect,
      mainContainer
    }

    mainG.append('g')
      .call(d3.axisLeft(yScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '11px')

    mainG.selectAll('.domain').attr('stroke', 'var(--border)')

    mainG.append('text')
      .attr('x', -mainHeight / 2)
      .attr('y', -42)
      .attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Avg Arrival Delay (min)')

    const brushSvg = d3.select(brushContainer)
      .append('svg')
      .attr('width', mainWidth + brushMargin.left + brushMargin.right)
      .attr('height', brushHeight + brushMargin.top + brushMargin.bottom)

    const brushG = brushSvg.append('g')
      .attr('transform', `translate(${brushMargin.left},${brushMargin.top})`)

    const yMinBrush = d3.min(fullData, d => d.actual) || 0
    const yMaxBrush = d3.max(fullData, d => d.actual) || 50
    const yScaleBrush = d3.scaleLinear()
      .domain([Math.min(0, yMinBrush), yMaxBrush])
      .range([brushHeight, 0])

    const brushLine = d3.line()
      .defined(d => d.actual != null)
      .x(d => xScaleFull(d.date))
      .y(d => yScaleBrush(d.actual))
      .curve(d3.curveMonotoneX)

    if (testStartDate) {
      brushG.append('line')
        .attr('x1', xScaleFull(testStartDate))
        .attr('x2', xScaleFull(testStartDate))
        .attr('y1', 0)
        .attr('y2', brushHeight)
        .attr('stroke', 'var(--red)')
        .attr('stroke-width', 1)
    }

    brushG.append('path')
      .datum(fullData)
      .attr('fill', 'none')
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-width', 1)
      .attr('d', brushLine)

    const updateMainChart = (newExtent) => {
      if (!chartRef.current) return
      const { xScaleMain, yScale, chartArea, xAxisG, lineActual, linePred, mainHeight, testStartDate: tsd, mainWidth } = chartRef.current

      xScaleMain.domain(newExtent)
      const visData = fullData.filter(d => d.date >= newExtent[0] && d.date <= newExtent[1])
      const yMinUpdate = d3.min(visData, d => Math.min(d.actual ?? 0, d.predicted ?? d.actual ?? 0)) || 0
      const yMax = d3.max(visData, d => Math.max(d.actual || 0, d.predicted || 0)) || 50
      yScale.domain([Math.min(0, yMinUpdate * 1.15), yMax * 1.15])

      chartRef.current.visibleData = visData

      chartArea.select('.line-actual').attr('d', lineActual(visData))
      const tData = visData.filter(d => d.isTest && d.predicted != null)
      chartArea.select('.line-pred').attr('d', linePred(tData))

      chartArea.select('.zero-line').remove()
      if (yMinUpdate < 0) {
        chartArea.append('line')
          .attr('class', 'zero-line')
          .attr('x1', 0)
          .attr('x2', mainWidth)
          .attr('y1', yScale(0))
          .attr('y2', yScale(0))
          .attr('stroke', 'var(--text-muted)')
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '4,4')
          .attr('opacity', 0.5)
      }

      xAxisG.call(d3.axisBottom(xScaleMain).ticks(6).tickFormat(formatAxisDate))
        .selectAll('text').style('fill', 'var(--text-muted)').style('font-size', '11px')

      const splitRect = chartArea.select('.train-region')
      const splitLine = chartArea.select('.split-line')
      if (tsd && tsd >= newExtent[0] && tsd <= newExtent[1]) {
        const sx = xScaleMain(tsd)
        splitRect.attr('width', sx).style('display', null)
        splitLine.attr('x1', sx).attr('x2', sx).style('display', null)
      } else {
        splitRect.style('display', 'none')
        splitLine.style('display', 'none')
      }
    }

    const brush = d3.brushX()
      .extent([[0, 0], [mainWidth, brushHeight]])
      .on('brush', (event) => {
        if (!event.selection || !event.sourceEvent) return
        const [x0, x1] = event.selection.map(xScaleFull.invert)
        updateMainChart([x0, x1])
      })
      .on('end', (event) => {
        if (!event.selection || !event.sourceEvent) return
        const [x0, x1] = event.selection.map(xScaleFull.invert)
        setBrushExtent([x0, x1])
      })

    const brushSelection = brushG.append('g')
      .attr('class', 'brush')
      .call(brush)

    brushRef.current = { brush, brushSelection, xScaleFull, updateMainChart }

    const initialSelection = [xScaleFull(currentExtent[0]), xScaleFull(currentExtent[1])]
    brushSelection.call(brush.move, initialSelection)

    brushG.append('g')
      .attr('transform', `translate(0,${brushHeight})`)
      .call(d3.axisBottom(xScaleFull).ticks(6).tickFormat(d3.timeFormat('%Y')))
      .selectAll('text')
      .style('fill', 'var(--text-muted)')
      .style('font-size', '10px')

    brushG.selectAll('.domain').attr('stroke', 'var(--border)')

  }, [fullData, dateExtent, testStartDate, defaultExtent, containerWidth])

  useEffect(() => {
    if (!brushRef.current || !brushExtent) return
    const { brush, brushSelection, xScaleFull, updateMainChart } = brushRef.current
    const newSelection = [xScaleFull(brushExtent[0]), xScaleFull(brushExtent[1])]
    brushSelection.call(brush.move, newSelection)
    if (updateMainChart) {
      updateMainChart(brushExtent)
    }
  }, [brushExtent])

  const modelName = forecastData?.models[selectedModel]?.name || selectedModel.toUpperCase()
  const currentModelDetails = MODEL_DETAILS[selectedModel] || MODEL_DETAILS.lstm
  const testPeriod = forecastData?.test_period
  const trainingPeriod = forecastData?.training_period

  if (loading) {
    return (
      <section id="forecast" className="section">
        <div className="container">
          <p className="kicker">Time Series Forecasting</p>
          <h2>Route Delay Forecasts</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading forecast data...
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section id="forecast" className="section">
        <div className="container">
          <p className="kicker">Time Series Forecasting</p>
          <h2>Route Delay Forecasts</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      </section>
    )
  }

  return (
    <section id="forecast" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Time Series Forecasting</p>
        <h2>Route Delay Forecasts</h2>
        <p style={{ marginBottom: 'var(--space-md)' }}>
          Next-day delay forecasts for the top 20 U.S. domestic routes reveal how models track real-world delay patterns over time. Each point represents the daily average arrival delay across all flights on a route, with actual delays in green and model predictions in dashed blue. Models were trained on data from January 2019 through December 2023 (left of the red split line), with January through June 2024 held out for tuning. The test period (right of the split) spans July 2024 through June 2025, representing true out-of-sample forecasts.
        </p>

        {currentModelDetails && (
          <div className="methodology-card" style={{ marginBottom: 'var(--space-lg)' }}>
            <button
              className="methodology-toggle"
              onClick={() => setShowMethodology(!showMethodology)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 0',
                color: 'var(--text-primary)',
                fontSize: '13px',
                fontWeight: '600'
              }}
            >
              <span style={{ transform: showMethodology ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>&#9654;</span>
              Model Details: {modelName}
              <span className={`model-detail-badge model-detail-badge--${currentModelDetails.type.toLowerCase().replace(' ', '-')}`}>
                {currentModelDetails.type}
              </span>
            </button>
            {showMethodology && (
              <div className="model-details-grid">
                <div className="model-detail-item">
                  <span className="model-detail-label">Architecture</span>
                  <span className="model-detail-value">{currentModelDetails.architecture}</span>
                </div>
                <div className="model-detail-item">
                  <span className="model-detail-label">Parameters</span>
                  <span className="model-detail-value">{currentModelDetails.params || 'N/A'}</span>
                </div>
                <div className="model-detail-item">
                  <span className="model-detail-label">Training</span>
                  <span className="model-detail-value">{currentModelDetails.training || 'N/A'}</span>
                </div>
                <div className="model-detail-item">
                  <span className="model-detail-label">Features</span>
                  <span className="model-detail-value">{currentModelDetails.features}</span>
                </div>
                <div className="model-detail-item">
                  <span className="model-detail-label">Test Period</span>
                  <span className="model-detail-value">{testPeriod?.start} to {testPeriod?.end}</span>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="viz-card" style={{ height: 'auto', padding: 0 }}>
          <div className="forecast-layout">
            <div className="forecast-controls">
              <h4>Controls</h4>
              <label>
                Route
                <select className="select" value={selectedCityPair || ''} onChange={(e) => setSelectedCityPair(e.target.value)}>
                  {cityPairs.map(pair => (
                    <option key={pair} value={pair}>{formatCityPairDisplay(pair)}</option>
                  ))}
                </select>
                <DirectionToggle
                  cityPair={selectedCityPair}
                  direction={direction}
                  onChange={setDirection}
                  availableRoutes={forecastData?.routes}
                />
              </label>
              <label>
                Model
                <select className="select" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                  {forecastData?.models && sortModelEntries(Object.entries(forecastData.models))
                    .map(([key, model]) => (
                      <option key={key} value={key}>{model.name}</option>
                    ))}
                </select>
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
                  <svg width="20" height="10"><line x1="10" y1="0" x2="10" y2="10" stroke="var(--red)" strokeWidth="2" strokeDasharray="3,2"/></svg>
                  <span>Train/Test Split</span>
                </div>
              </div>
            </div>

            <div className="forecast-chart" ref={chartAreaRef}>
              <div ref={mainChartRef} style={{ width: '100%', height: '320px' }} />
              <div className="forecast-brush-row">
                <div ref={brushChartRef} style={{ flex: 1, height: '60px' }} />
              </div>
              <div className="forecast-chart-actions">
                <button className="chart-btn" onClick={handleZoomIn} title="Zoom In">+</button>
                <button className="chart-btn" onClick={handleZoomOut} title="Zoom Out">-</button>
                <button className="chart-btn-text" onClick={handleReset}>Reset View</button>
                <button className="chart-btn-text" onClick={handleShowAll}>Show All</button>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 'var(--space-xs)' }}>
                Drag the highlighted region to pan through time, or drag its edges to resize the window.
              </p>
            </div>

            <div className="forecast-summary" style={{ justifyContent: 'center' }}>
              <div style={{ marginBottom: 'var(--space-md)', marginTop: 'calc(-1 * var(--space-lg))', marginLeft: 'calc(-1 * var(--space-lg))', marginRight: 'calc(-1 * var(--space-lg))', padding: 'var(--space-sm) var(--space-lg)', background: 'var(--bg-base-soft)', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Test Period (Jul 2024 - Jun 2025)
                </span>
              </div>

              <div className="summary-header">
                <span className="summary-header__model">{modelName}</span>
                <span className="summary-header__route">{selectedRoute}</span>
              </div>

              <h4>Route Metrics</h4>
              <div className="forecast-summary-metrics">
                <div className="metric-row">
                  <span className="metric-row__label">
                    <span className="metric-row__dot metric-row__dot--mae"></span>
                    MAE
                  </span>
                  <span className="metric-row__value">{metrics?.route?.mae ? `${metrics.route.mae.toFixed(2)} min` : '-'}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-row__label">
                    <span className="metric-row__dot metric-row__dot--rmse"></span>
                    RMSE
                  </span>
                  <span className="metric-row__value">{metrics?.route?.rmse ? `${metrics.route.rmse.toFixed(2)} min` : '-'}</span>
                </div>
                <div className="metric-row">
                  <span
                    className="metric-row__label"
                    style={{ cursor: 'help' }}
                    onMouseEnter={(e) => showTooltip(e, 'Forecasts within ±15 minutes of actual daily average')}
                    onMouseLeave={hideTooltip}
                  >
                    <span className="metric-row__dot metric-row__dot--within15"></span>
                    Hit Rate
                  </span>
                  <span className="metric-row__value">{metrics?.route?.within_15 ? `${metrics.route.within_15.toFixed(2)}%` : '-'}</span>
                </div>
              </div>

              {metrics?.overall && (
                <div className="summary-section">
                  <div className="summary-section__title">Overall Test Metrics</div>
                  <div className="forecast-summary-metrics">
                    <div className="metric-row">
                      <span className="metric-row__label">
                        <span className="metric-row__dot metric-row__dot--mae"></span>
                        MAE
                      </span>
                      <span className="metric-row__value">{metrics.overall.mae?.toFixed(2)} min</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-row__label">
                        <span className="metric-row__dot metric-row__dot--rmse"></span>
                        RMSE
                      </span>
                      <span className="metric-row__value">{metrics.overall.rmse?.toFixed(2)} min</span>
                    </div>
                    <div className="metric-row">
                      <span
                        className="metric-row__label"
                        style={{ cursor: 'help' }}
                        onMouseEnter={(e) => showTooltip(e, 'Forecasts within ±15 minutes of actual daily average')}
                        onMouseLeave={hideTooltip}
                      >
                        <span className="metric-row__dot metric-row__dot--within15"></span>
                        Hit Rate
                      </span>
                      <span className="metric-row__value">{metrics.overall.within_15?.toFixed(2)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ForecastChart
