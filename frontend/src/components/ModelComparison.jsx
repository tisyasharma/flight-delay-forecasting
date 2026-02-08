import { useMemo } from 'react'
import { MODEL_COLORS } from '../utils/constants.js'

function ModelComparison({ forecastData, loading, error }) {
  const modelResults = useMemo(() => {
    if (!forecastData?.walk_forward?.models) return []

    const wfModels = forecastData.walk_forward.models
    const modelNameMap = {
      naive: 'Naive',
      ma: 'Moving Average',
      xgboost: 'XGBoost',
      lightgbm: 'LightGBM',
      lstm: 'LSTM',
      tcn: 'TCN'
    }

    return Object.entries(wfModels).map(([key, metrics]) => ({
      model: modelNameMap[key] || key,
      mae: metrics.mae.mean,
      within_15: metrics.within_15.mean,
      rmse: metrics.rmse.mean,
      median_ae: metrics.median_ae.mean,
      r2: metrics.r2.mean,
    }))
  }, [forecastData])

  if (loading) {
    return (
      <section id="model-comparison" className="section">
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
      <section id="model-comparison" className="section">
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
      <section id="model-comparison" className="section">
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
      ? `${value.toFixed(2)}%`
      : `${value.toFixed(2)} min`

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
    <section id="model-comparison" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Performance</p>
        <h2>Model Comparison</h2>
        <p style={{ marginBottom: 'var(--space-xl)' }}>
          We evaluate four modeling approaches for next-day route-level delay forecasting using walk-forward cross-validation (2023-2024). The comparison includes two gradient boosting models (XGBoost, LightGBM) and two deep learning models (LSTM, TCN). Performance is assessed using MAE and hit rate, defined as the share of daily forecasts within ±15 minutes of the observed delay. Metrics are aggregated across all 50 training routes.
        </p>

        <div className="viz-card model-comparison-card" style={{ height: 'auto', padding: 0 }}>
          <div style={{ padding: 'var(--space-sm) var(--space-lg)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Walk-Forward Validation (2023-2024)
            </span>
          </div>
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
            <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span style={{ fontWeight: 600 }}>Notes on Evaluation:</span> Results reflect a single training run (seed = 42). Differences under 0.5 percentage points fall within typical run-to-run variation and can be interpreted as effectively tied.
            </p>
          </div>
        </div>

        <div className="findings-grid findings-grid--3">
          <div className="finding-card finding-card--green">
            <h4>Best Performance</h4>
            <p>Gradient boosting achieved the strongest overall performance, with 77.7% of forecasts falling within ±15 minutes of the observed delay and a 25% reduction in MAE relative to a naive baseline.</p>
          </div>
          <div className="finding-card finding-card--cyan">
            <h4>Gradient Boosting vs Deep Learning</h4>
            <p>On this dataset, gradient boosting outperformed deep learning by 3.3 percentage points on hit rate (77.7% vs 74.4%). Within each model family, performance differences were small: XGBoost and LightGBM tied at 11.25 min MAE, while LSTM and TCN clustered near 12.7 min MAE.</p>
          </div>
          <div className="finding-card finding-card--orange">
            <h4>Why Features Matter</h4>
            <p>In ablation testing, gradient boosting without weather features still outperformed deep learning with the full feature set (12.69 min MAE). Weather features alone accounted for a 10.3% improvement in XGBoost performance.</p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ModelComparison
