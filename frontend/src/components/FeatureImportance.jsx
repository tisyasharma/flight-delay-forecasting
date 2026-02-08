import { useEffect, useMemo, useState } from 'react'
import { assetUrl } from '../utils/helpers.js'

const CATEGORY_CONFIG = {
  lag: { label: 'Recent Delays', color: '#1e40af' },
  weather: { label: 'Weather', color: '#513288' },
  temporal: { label: 'Temporal', color: '#842362' },
  route: { label: 'Route', color: '#c8102e' }
}

const FEATURE_MAPPING = {
  // Lag/Rolling delay features
  'ewm_7_arr_delay': { label: 'EWM Delay (7d)', category: 'lag' },
  'lag_1_arr_delay': { label: 'Delay Lag 1d', category: 'lag' },
  'lag_7_arr_delay': { label: 'Delay Lag 7d', category: 'lag' },
  'lag_14_arr_delay': { label: 'Delay Lag 14d', category: 'lag' },
  'lag_28_arr_delay': { label: 'Delay Lag 28d', category: 'lag' },
  'rolling_mean_7_arr_delay': { label: 'Rolling Mean 7d', category: 'lag' },
  'rolling_mean_14_arr_delay': { label: 'Rolling Mean 14d', category: 'lag' },
  'rolling_std_7_arr_delay': { label: 'Rolling Std 7d', category: 'lag' },
  'rolling_std_14_arr_delay': { label: 'Rolling Std 14d', category: 'lag' },

  // Weather features
  'apt1_severity': { label: 'Airport 1 Weather', category: 'weather' },
  'apt2_severity': { label: 'Airport 2 Weather', category: 'weather' },
  'weather_severity_max': { label: 'Max Weather Severity', category: 'weather' },
  'weather_severity_combined': { label: 'Combined Weather', category: 'weather' },
  'has_adverse_weather': { label: 'Adverse Weather', category: 'weather' },
  'severe_weather_level': { label: 'Severe Weather', category: 'weather' },
  'both_clear': { label: 'Both Clear', category: 'weather' },
  'weather_diff': { label: 'Weather Difference', category: 'weather' },
  'apt1_temp_avg': { label: 'Airport 1 Temp', category: 'weather' },
  'apt2_temp_avg': { label: 'Airport 2 Temp', category: 'weather' },
  'apt1_precip_total': { label: 'Airport 1 Precip', category: 'weather' },
  'apt2_precip_total': { label: 'Airport 2 Precip', category: 'weather' },
  'apt1_snowfall': { label: 'Airport 1 Snow', category: 'weather' },
  'apt2_snowfall': { label: 'Airport 2 Snow', category: 'weather' },
  'apt1_wind_speed_max': { label: 'Airport 1 Wind', category: 'weather' },
  'apt2_wind_speed_max': { label: 'Airport 2 Wind', category: 'weather' },
  'total_precip': { label: 'Total Precipitation', category: 'weather' },
  'total_snowfall': { label: 'Total Snowfall', category: 'weather' },
  'max_wind': { label: 'Max Wind Speed', category: 'weather' },
  'weather_severity_lag_1': { label: 'Weather Lag 1d', category: 'weather' },
  'weather_severity_lag_3': { label: 'Weather Lag 3d', category: 'weather' },
  'weather_severity_lag_7': { label: 'Weather Lag 7d', category: 'weather' },
  // Hourly-derived weather features
  'peak_wind_operating': { label: 'Peak Wind (Operating Hours)', category: 'weather' },
  'precip_operating': { label: 'Precip (Operating Hours)', category: 'weather' },
  'max_hourly_severity': { label: 'Max Hourly Severity', category: 'weather' },
  'storm_hours': { label: 'Storm Hours', category: 'weather' },
  'morning_severity': { label: 'Morning Severity', category: 'weather' },
  'evening_severity': { label: 'Evening Severity', category: 'weather' },
  // Temporal features
  'is_weekend': { label: 'Weekend', category: 'temporal' },
  'is_summer': { label: 'Summer Season', category: 'temporal' },
  'is_winter': { label: 'Winter Season', category: 'temporal' },
  'quarter': { label: 'Quarter', category: 'temporal' },
  'day_sin': { label: 'Annual Cycle (sin)', category: 'temporal' },
  'day_cos': { label: 'Annual Cycle (cos)', category: 'temporal' },
  'month_sin': { label: 'Month Cycle (sin)', category: 'temporal' },
  'month_cos': { label: 'Month Cycle (cos)', category: 'temporal' },
  'day_of_week': { label: 'Day of Week', category: 'temporal' },
  'day_of_week_sin': { label: 'Weekly Cycle (sin)', category: 'temporal' },
  'day_of_week_cos': { label: 'Weekly Cycle (cos)', category: 'temporal' },
  'month': { label: 'Month', category: 'temporal' },
  'week_of_year': { label: 'Week of Year', category: 'temporal' },
  'day_of_month': { label: 'Day of Month', category: 'temporal' },
  'is_month_end': { label: 'Month End', category: 'temporal' },
  'is_month_start': { label: 'Month Start', category: 'temporal' },
  'is_covid_period': { label: 'COVID Period', category: 'temporal' },
  'is_post_covid': { label: 'Post-COVID Era', category: 'temporal' },
  'is_covid_recovery': { label: 'COVID Recovery', category: 'temporal' },
  'is_school_break': { label: 'School Break', category: 'temporal' },
  'is_federal_holiday': { label: 'Federal Holiday', category: 'temporal' },
  'is_holiday_week': { label: 'Holiday Week', category: 'temporal' },
  'days_from_holiday': { label: 'Days from Holiday', category: 'temporal' },
  'days_to_holiday': { label: 'Days to Holiday', category: 'temporal' },

  // Route features
  'route_avg_delay': { label: 'Route History', category: 'route' },
  'route_delay_std': { label: 'Route Delay Std', category: 'route' },
  'route_delay_mean': { label: 'Route Delay Avg', category: 'route' },
  'route_std_demand': { label: 'Route Demand Std', category: 'route' },
  'route_mean_demand': { label: 'Route Demand Avg', category: 'route' },
  'route_median_demand': { label: 'Route Demand Median', category: 'route' },
  'is_hub_dest': { label: 'Hub Destination', category: 'route' },
  'is_hub_origin': { label: 'Hub Origin', category: 'route' },
  'DISTANCE': { label: 'Distance', category: 'route' },
  'dest_encoded': { label: 'Destination', category: 'route' },
  'origin_encoded': { label: 'Origin', category: 'route' },
  'route_encoded': { label: 'Route', category: 'route' },

  // Congestion features (legacy mapping)
  'dest_rolling_delay_7d': { label: 'Dest Rolling Delay', category: 'lag' },
  'origin_rolling_delay_7d': { label: 'Origin Rolling Delay', category: 'lag' },
  'dest_daily_flights': { label: 'Dest Volume', category: 'route' },
  'origin_daily_flights': { label: 'Origin Volume', category: 'route' }
}

function FeatureImportance() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [selectedCategory, setSelectedCategory] = useState(null)
  const [expandedCategories, setExpandedCategories] = useState(new Set(['lag', 'weather', 'temporal', 'route', 'carrier']))

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      const response = await fetch(assetUrl('data/feature_importance.json'))
      if (!response.ok) throw new Error('Failed to load feature importance data')
      const jsonData = await response.json()
      setData(jsonData)
    } catch (err) {
      setError('Failed to load feature importance data')
    } finally {
      setLoading(false)
    }
  }

  const availableModels = useMemo(() => {
    if (!data?.models) return []
    return Object.entries(data.models).map(([key, model]) => ({
      key,
      name: model.name
    }))
  }, [data])

  const processedData = useMemo(() => {
    if (!data?.models?.[selectedModel]) return { features: [], categoryTotals: {}, maxImportance: 0 }

    const modelFeatures = data.models[selectedModel].features
    const features = modelFeatures.map((f, idx) => {
      const mapping = FEATURE_MAPPING[f.feature] || { label: f.feature, category: 'other' }
      return {
        ...f,
        label: mapping.label,
        category: mapping.category,
        color: CATEGORY_CONFIG[mapping.category]?.color || '#6b7280',
        rank: idx + 1
      }
    })

    const categoryTotals = {}
    features.forEach(f => {
      if (!categoryTotals[f.category]) categoryTotals[f.category] = 0
      categoryTotals[f.category] += f.importance
    })

    const maxImportance = Math.max(...features.map(f => f.importance))

    return { features, categoryTotals, maxImportance }
  }, [data, selectedModel])

  const groupedFeatures = useMemo(() => {
    const groups = {}
    processedData.features.forEach(f => {
      if (!groups[f.category]) groups[f.category] = []
      groups[f.category].push(f)
    })
    return groups
  }, [processedData.features])

  const filteredFeatures = useMemo(() => {
    if (!selectedCategory) return processedData.features
    return processedData.features.filter(f => f.category === selectedCategory)
  }, [processedData.features, selectedCategory])

  const toggleCategory = (cat) => {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(cat)) {
        next.delete(cat)
      } else {
        next.add(cat)
      }
      return next
    })
  }

  if (loading) {
    return (
      <section id="feature-importance" className="section">
        <div className="container">
          <p className="kicker">Feature Evaluation</p>
          <h2>Key Contributors to Route Delay Forecasts</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading feature data...
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section id="feature-importance" className="section">
        <div className="container">
          <p className="kicker">Feature Evaluation</p>
          <h2>Key Contributors to Route Delay Forecasts</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      </section>
    )
  }

  const categoryOrder = ['lag', 'weather', 'temporal', 'route']

  return (
    <section id="feature-importance" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Feature Evaluation</p>
        <h2>Key Contributors to Route Delay Forecasts</h2>
        <p style={{ marginBottom: 'var(--space-lg)' }}>
          Feature importance highlights which of the 63 input variables contribute most to model performance when forecasting delays across 50 U.S. domestic routes (January 2019 through December 2023 training period). Importance rankings may vary slightly between XGBoost and LightGBM due to model-specific calculation methods.
        </p>

        <div className="feature-layout">
          <div className="feature-layout__main">
            <div className="viz-card" style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 'var(--space-md) var(--space-lg)', borderBottom: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {data?.models?.[selectedModel]?.name} Feature Importance
                </span>
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
              <div className="feature-table">
                <div className="feature-table__header">
                  <div className="feature-table__col feature-table__col--name">Feature</div>
                  <div className="feature-table__col feature-table__col--category">Category</div>
                  <div className="feature-table__col feature-table__col--importance">Importance</div>
                  <div className="feature-table__col feature-table__col--bar"></div>
                </div>

                {categoryOrder.map(cat => {
                  const features = groupedFeatures[cat] || []
                  if (features.length === 0) return null
                  const config = CATEGORY_CONFIG[cat]
                  const isExpanded = expandedCategories.has(cat)
                  const categoryTotal = processedData.categoryTotals[cat] || 0

                  return (
                    <div key={cat} className="feature-table__group">
                      <div
                        className="feature-table__group-header"
                        onClick={() => toggleCategory(cat)}
                        style={{ borderLeftColor: config.color }}
                      >
                        <div className="feature-table__group-toggle">
                          <span className={`feature-table__chevron ${isExpanded ? 'expanded' : ''}`}>
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                              <path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </span>
                          <span className="feature-table__group-name">{config.label}</span>
                          <span className="feature-table__group-count">{features.length} features</span>
                        </div>
                        <div className="feature-table__group-total">
                          <span className="feature-table__group-pct">{(categoryTotal * 100).toFixed(1)}%</span>
                          <div className="feature-table__group-bar">
                            <div
                              className="feature-table__group-bar-fill"
                              style={{
                                width: `${categoryTotal * 100}%`,
                                backgroundColor: config.color
                              }}
                            />
                          </div>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="feature-table__rows">
                          {features.map((f, idx) => (
                            <div
                              key={f.feature}
                              className="feature-table__row"
                              style={{ animationDelay: `${idx * 30}ms` }}
                            >
                              <div className="feature-table__col feature-table__col--name">
                                {f.label}
                              </div>
                              <div className="feature-table__col feature-table__col--category">
                                <span
                                  className="feature-table__tag"
                                  style={{ backgroundColor: `${config.color}30`, color: config.color }}
                                >
                                  {config.label}
                                </span>
                              </div>
                              <div className="feature-table__col feature-table__col--importance">
                                <span className="feature-table__value">{(f.importance * 100).toFixed(2)}%</span>
                              </div>
                              <div className="feature-table__col feature-table__col--bar">
                                <div className="feature-table__bar">
                                  <div
                                    className="feature-table__bar-fill"
                                    style={{
                                      width: `${f.importance * 100}%`,
                                      backgroundColor: config.color
                                    }}
                                  />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
              <p style={{ margin: 0, padding: 'var(--space-sm) var(--space-lg)', fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border)', background: 'var(--bg-base-soft)' }}>
                Feature importance reflects how the selected model uses inputs to reduce forecast error, not causal effects on delays.
              </p>
            </div>
          </div>

          <div className="feature-layout__sidebar">
            {(() => {
              const sorted = [...categoryOrder].sort((a, b) =>
                (processedData.categoryTotals[b] || 0) - (processedData.categoryTotals[a] || 0)
              )
              const topCategory = sorted[0]
              const bottomCategory = sorted[sorted.length - 1]

              const categoryDescriptions = {
                weather: (cat) => {
                  if (cat === topCategory) return 'Weather features collectively account for the largest share of importance, indicating strong sensitivity to adverse conditions and precipitation in the model.'
                  return 'Weather features capture adverse conditions and precipitation effects on delays.'
                },
                lag: () => 'Recent delay history provides strong momentum signals for next-day route forecasts.',
                temporal: () => 'Calendar-based features, including COVID-era shifts, seasonality, and holidays, capture recurring patterns that influence forecast accuracy.',
                route: (cat) => {
                  if (cat === bottomCategory) return 'Route-level identifiers contribute the least incremental predictive value, suggesting limited additional signal beyond weather, temporal, and recent-delay features.'
                  return 'Route-level features capture corridor-specific characteristics like distance and hub status.'
                }
              }

              return sorted.map(cat => {
                const config = CATEGORY_CONFIG[cat]
                const pct = ((processedData.categoryTotals[cat] || 0) * 100).toFixed(1)
                const catFeatures = groupedFeatures[cat] || []
                const topFeature = catFeatures[0]

                return (
                  <div key={cat} className="finding-card">
                    <h4>{config.label} ({pct}%)</h4>
                    <p>
                      {topFeature
                        ? `Top feature: ${topFeature.label} (${(topFeature.importance * 100).toFixed(2)}%). `
                        : ''}
                      {categoryDescriptions[cat]?.(cat) || ''}
                    </p>
                  </div>
                )
              })
            })()}
          </div>
        </div>
      </div>
    </section>
  )
}

export default FeatureImportance
