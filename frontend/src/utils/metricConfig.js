// centralized metric config so all components use the same labels, units, and colors

export const METRIC_CONFIG = {
  mae: {
    label: 'MAE',
    unit: 'min',
    color: '#EF4444',
    tier: 1,
    description: 'Mean Absolute Error - average prediction error'
  },
  rmse: {
    label: 'RMSE',
    unit: 'min',
    color: '#38BDF8',
    tier: 1,
    description: 'Root Mean Squared Error - penalizes large errors'
  },
  within_15: {
    label: 'Hit Rate',
    unit: '%',
    color: '#10b981',
    tier: 1,
    description: 'Forecasts within ±15 minutes of actual daily average'
  },
  threshold_acc: {
    label: 'Delay Detection',
    unit: '%',
    color: '#6366F1',
    tier: 2,
    description: 'Correctly classified delay days (avg >15 min) vs normal days'
  },
  r2: {
    label: 'R\u00B2',
    unit: '',
    color: '#6366F1',
    tier: 2,
    description: 'Coefficient of Determination (variance explained)'
  },
  median_ae: {
    label: 'Median Error',
    unit: 'min',
    color: '#FBBF24',
    tier: 2,
    description: 'Median Absolute Error - robust to outliers'
  },
}

export const PRIMARY_METRICS = ['mae', 'rmse', 'within_15']

export function formatMetricValue(key, value) {
  if (value == null) return '-'
  const config = METRIC_CONFIG[key]
  if (!config) return value

  if (key === 'r2') return value.toFixed(3)
  if (config.unit === '%') return `${value.toFixed(1)}%`
  return `${value.toFixed(1)} ${config.unit}`
}

export function getMetricLabel(key) {
  return METRIC_CONFIG[key]?.label || key
}

export function getMetricColor(key) {
  return METRIC_CONFIG[key]?.color || '#666666'
}
