import { useMemo } from 'react'
import { MODEL_COLORS } from '../utils/constants.js'

function ModelComparison({ forecastData, loading, error }) {
  const modelResults = useMemo(() => {
    if (!forecastData?.models) return []
    return Object.entries(forecastData.models).map(([key, model]) => ({
      model: model.name,
      mae: model.metrics.overall.mae,
      within_15: model.metrics.overall.within_15,
      rmse: model.metrics.overall.rmse,
      median_ae: model.metrics.overall.median_ae,
      threshold_acc: model.metrics.overall.threshold_acc,
      r2: model.metrics.overall.r2,
    }))
  }, [forecastData])

  if (loading) {
    return (
      <section id="model-comparison" className="section section--alt">
        <div className="container">
          <p className="kicker">Performance</p>
          <h2>Model Comparison</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading model comparison data...
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section id="model-comparison" className="section section--alt">
        <div className="container">
          <p className="kicker">Performance</p>
          <h2>Model Comparison</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      </section>
    )
  }

  const baseline = modelResults.find(m => m.model === 'Naive')
  const BASELINE_NAMES = ['Naive', 'Moving Average']
  const mlModels = modelResults.filter(m => !BASELINE_NAMES.includes(m.model))

  if (mlModels.length === 0) {
    return (
      <section id="model-comparison" className="section section--alt">
        <div className="container">
          <p className="kicker">Performance</p>
          <h2>Model Comparison</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading model comparison data...
          </div>
        </div>
      </section>
    )
  }

  const sortedByMAE = [...mlModels].sort((a, b) => a.mae - b.mae)
  const sortedByWithin15 = [...mlModels].sort((a, b) => b.within_15 - a.within_15)

  const bestMAE = sortedByMAE[0]
  const bestWithin15 = sortedByWithin15[0]

  const MAE_THRESHOLD = 0.2
  const HIT_RATE_THRESHOLD = 0.5

  const maeGap = sortedByMAE.length > 1 ? sortedByMAE[1].mae - bestMAE.mae : 0
  const hitRateGap = sortedByWithin15.length > 1 ? bestWithin15.within_15 - sortedByWithin15[1].within_15 : 0

  const showMAEBest = maeGap >= MAE_THRESHOLD
  const showHitRateBest = hitRateGap >= HIT_RATE_THRESHOLD

  const improvementPct = baseline
    ? ((baseline.mae - bestMAE.mae) / baseline.mae * 100).toFixed(1)
    : '0'

  const maxMAE = Math.max(...mlModels.map(m => m.mae))
  const maxWithin15 = 100

  const getBarWidth = (value, max) => `${(value / max) * 100}%`

  const renderHorizontalBar = (item, metric, max, isBest, showBestBadge) => {
    const value = item[metric]
    const displayValue = metric === 'within_15'
      ? `${value.toFixed(1)}%`
      : `${value.toFixed(1)} min`

    const barColor = MODEL_COLORS[item.model] || '#64748b'

    return (
      <div key={item.model} className="model-bar">
        <span className="model-bar__name">{item.model}</span>
        <div className="model-bar__track">
          <div
            className="model-bar__fill"
            style={{
              width: getBarWidth(Math.abs(value), max),
              backgroundColor: barColor
            }}
          />
        </div>
        <span className="model-bar__value">{displayValue}</span>
        <span className="model-bar__badge-container">
          {isBest && showBestBadge && <span className="model-bar__badge model-bar__badge--best">BEST</span>}
        </span>
      </div>
    )
  }

  return (
    <section id="model-comparison" className="section section--alt">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Performance</p>
        <h2>Model Comparison</h2>
        <p style={{ marginBottom: 'var(--space-xl)' }}>
          Four machine learning approaches compared on forecast error (MAE, lower is better) and hit rate, the percentage of daily route-level forecasts falling within a ±15-minute threshold. Two gradient boosting models (XGBoost, LightGBM) and two deep learning models (LSTM, TCN) are evaluated on the held-out test period (July 2024 through June 2025). Overall metrics are computed across all 50 training routes. Baselines (naive lag-1 at 14.9 min MAE, 7-day moving average at 13.6 min MAE) are excluded from the visualization but inform the improvement calculations below.
        </p>

        <div className="viz-card model-comparison-card" style={{ height: 'auto', padding: 0 }}>
          <div style={{ padding: 'var(--space-md) var(--space-lg)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
            <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              MAE (Mean Absolute Error)
            </h4>
          </div>
          <div style={{ padding: 'var(--space-lg)', background: 'var(--bg-base-elevated)' }}>
            {sortedByMAE.map(item =>
              renderHorizontalBar(
                item,
                'mae',
                maxMAE,
                item.mae === bestMAE.mae && sortedByMAE.filter(m => m.mae === bestMAE.mae).length === 1,
                showMAEBest
              )
            )}
          </div>

          <div style={{ padding: 'var(--space-md) var(--space-lg)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
            <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Hit Rate (±15 Minutes)
            </h4>
          </div>
          <div style={{ padding: 'var(--space-lg)', background: 'var(--bg-base-elevated)' }}>
            {sortedByWithin15.map(item =>
              renderHorizontalBar(
                item,
                'within_15',
                maxWithin15,
                item.within_15 === bestWithin15.within_15 && sortedByWithin15.filter(m => m.within_15 === bestWithin15.within_15).length === 1,
                showHitRateBest
              )
            )}
          </div>

          <div style={{ padding: 'var(--space-sm) var(--space-lg)', background: 'var(--bg-base-soft)', borderTop: '1px solid var(--border)' }}>
            <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Results from a single training run (seed=42). Differences under 0.5 percentage points are within typical random variation.
            </p>
          </div>
        </div>

        <div className="findings-grid findings-grid--3">
          <div className="finding-card finding-card--green">
            <h4>Best Performance</h4>
            <p>{(() => {
              if (!baseline || !bestMAE) return 'Best models significantly outperform baselines.'
              const improvement = ((baseline.mae - bestMAE.mae) / baseline.mae * 100).toFixed(0)
              return `${bestWithin15.within_15.toFixed(1)}% of gradient boosting predictions land within 15 minutes of actual delays. This represents a ${improvement}% reduction in forecast error compared to the naive baseline.`
            })()}</p>
          </div>
          <div className="finding-card finding-card--cyan">
            <h4>Gradient Boosting vs Deep Learning</h4>
            <p>{(() => {
              const xgb = modelResults.find(m => m.model === 'XGBoost')
              const lstm = modelResults.find(m => m.model === 'LSTM')
              if (!xgb || !lstm) return 'Gradient boosting outperforms deep learning.'
              const hitRateGap = (bestWithin15.within_15 - lstm.within_15).toFixed(1)
              return `On this dataset, gradient boosting outperformed deep learning by ${hitRateGap} percentage points on hit rate (${bestWithin15.within_15.toFixed(1)}% vs ${lstm.within_15.toFixed(1)}%). Within each category, algorithm choice barely mattered: XGBoost and LightGBM tied at ${xgb.mae.toFixed(1)} min MAE, and LSTM and TCN tied at ${lstm.mae.toFixed(1)} min.`
            })()}</p>
          </div>
          <div className="finding-card finding-card--orange">
            <h4>Why Features Matter</h4>
            <p>{(() => {
              const lstm = modelResults.find(m => m.model === 'LSTM')
              if (!lstm) return 'Feature engineering drives model performance.'
              return `In our ablation study, gradient boosting without weather data still outperformed deep learning with the full feature set (${lstm.mae.toFixed(1)} min MAE). Weather features alone accounted for a 10.3% improvement in XGBoost performance.`
            })()}</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ModelComparison
