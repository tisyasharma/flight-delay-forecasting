import { useEffect, useMemo, useState } from 'react'
import { assetUrl } from '../utils/helpers.js'

const CARRIER_CONFIG = {
  'AA': { name: 'American Airlines', color: '#0078D2' },
  'DL': { name: 'Delta Air Lines', color: '#E01933' },
  'UA': { name: 'United Airlines', color: '#002244' },
  'WN': { name: 'Southwest Airlines', color: '#FFBF27' },
  'B6': { name: 'JetBlue Airways', color: '#0033A1' },
  'AS': { name: 'Alaska Airlines', color: '#01426A' },
  'NK': { name: 'Spirit Airlines', color: '#FFE31A' },
  'F9': { name: 'Frontier Airlines', color: '#0A5640' },
  'G4': { name: 'Allegiant Air', color: '#F57F29' },
  'HA': { name: 'Hawaiian Airlines', color: '#9E0B3D' }
}

function CarrierPerformance() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState('on_time_pct')
  const [sortDir, setSortDir] = useState('desc')
  const [showTooltip, setShowTooltip] = useState(false)
  const [showModelingNote, setShowModelingNote] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const [error, setError] = useState(null)

  const loadData = async () => {
    try {
      setLoading(true)
      const response = await fetch(assetUrl('data/carrier_performance.json'))
      if (!response.ok) throw new Error('Failed to load carrier performance data')
      const jsonData = await response.json()
      setData(jsonData)
    } catch (err) {
      setError('Failed to load carrier performance data')
    } finally {
      setLoading(false)
    }
  }

  const sortedData = useMemo(() => {
    if (!data) return []
    const sorted = [...data.carriers].sort((a, b) => {
      const aVal = a[sortBy]
      const bVal = b[sortBy]
      return sortDir === 'desc' ? bVal - aVal : aVal - bVal
    })
    return sorted
  }, [data, sortBy, sortDir])

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortDir(prev => prev === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(field)
      setSortDir(field === 'avg_delay' || field === 'severe_delay_pct' ? 'asc' : 'desc')
    }
  }

  const maxDelay = useMemo(() => {
    if (!data) return 1
    return Math.max(...data.carriers.map(c => c.avg_delay))
  }, [data])

  const getPerformanceColor = (onTimePct) => {
    const minPct = 0.65
    const maxPct = 0.88
    const normalized = Math.max(0, Math.min(1, (onTimePct - minPct) / (maxPct - minPct)))

    if (normalized <= 0.5) {
      const t = normalized / 0.5
      const r = Math.round(239 + (245 - 239) * t)
      const g = Math.round(68 + (158 - 68) * t)
      const b = Math.round(68 + (11 - 68) * t)
      return `rgb(${r}, ${g}, ${b})`
    } else {
      const t = (normalized - 0.5) / 0.5
      const r = Math.round(245 + (34 - 245) * t)
      const g = Math.round(158 + (197 - 158) * t)
      const b = Math.round(11 + (94 - 11) * t)
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  const SortIcon = ({ field }) => {
    const isActive = sortBy === field
    return (
      <span className={`carrier-table__sort-icon ${isActive ? 'active' : ''}`}>
        {isActive && sortDir === 'desc' ? (
          <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
            <path d="M5 12L0 6h10L5 12z"/>
          </svg>
        ) : isActive && sortDir === 'asc' ? (
          <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
            <path d="M5 0l5 6H0l5-6z"/>
          </svg>
        ) : (
          <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
            <path d="M5 0l4 5H1l4-5zM5 12L1 7h8l-4 5z"/>
          </svg>
        )}
      </span>
    )
  }

  if (loading) {
    return (
      <section id="carrier-performance" className="section">
        <div className="container">
          <p className="kicker">Industry Context</p>
          <h2>Carrier Performance Rankings</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
            Loading carrier data...
          </div>
        </div>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section id="carrier-performance" className="section">
        <div className="container">
          <p className="kicker">Industry Context</p>
          <h2>Carrier Performance Rankings</h2>
          <div className="viz-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '400px', color: 'var(--red)' }}>
            {error || 'Failed to load carrier performance data'}
          </div>
        </div>
      </section>
    )
  }

  const dateRange = data?.date_range
  const formatDateRange = (dateStr) => {
    const [year, month] = dateStr.split('-')
    const date = new Date(year, month - 1)
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
  }
  const dateRangeText = dateRange
    ? `${formatDateRange(dateRange.start)} to ${formatDateRange(dateRange.end)}`
    : 'January 2019 to June 2025'

  return (
    <section id="carrier-performance" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Industry Context</p>
        <h2>Carrier Performance Rankings</h2>
        <p style={{ marginBottom: 'var(--space-sm)' }}>
          Airline choice is associated with measurable differences in delay outcomes, even when aggregated at the route level. To provide industry context for our forecasts, we compare on-time performance across the top 10 major U.S. carriers from {dateRangeText}.<sup><a href="#ref-2">2</a></sup> Regional operators are excluded, as their operations are conducted under major airlines and are not independently scheduled.
        </p>
        <p style={{ marginBottom: showModelingNote ? 'var(--space-sm)' : 'var(--space-lg)' }}>
          Across all metrics examined, premium and legacy carriers consistently outperform low-cost alternatives in on-time arrivals, average delay, and frequency of severe disruptions.{' '}
          <span
            onClick={() => setShowModelingNote(!showModelingNote)}
            style={{
              color: 'var(--accent)',
              cursor: 'pointer',
              fontWeight: 500,
              whiteSpace: 'nowrap'
            }}
          >
            {showModelingNote ? 'Show less' : 'View modeling note...'}
          </span>
        </p>
        {showModelingNote && (
          <div
            style={{
              marginBottom: 'var(--space-lg)',
              color: 'var(--text-secondary)',
              paddingLeft: 'var(--space-md)',
              borderLeft: '2px solid var(--border)'
            }}
          >
            <p style={{ marginBottom: 'var(--space-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>Modeling Implications</p>
            <p style={{ marginBottom: 'var(--space-sm)' }}>
              Airline identity is not included as a model feature, and forecasts are generated using route-level averages across all carriers. Carrier-specific operational strategies and performance differences are therefore not explicitly modeled and are largely averaged out in the target variable.
            </p>
            <p>
              These rankings are presented as industry context rather than predictive signal. The model's objective is to forecast overall delay risk on a route for a given day, reflecting system-level conditions such as weather, congestion, and seasonal demand that affect all operators on a corridor.
            </p>
          </div>
        )}

        <div className="carrier-layout">
          <div className="carrier-layout__main">
            <div className="viz-card" style={{ padding: 0, height: 'auto' }}>
              <div className="carrier-table">
                <div className="carrier-table__header">
                  <div className="carrier-table__col carrier-table__col--rank">Rank</div>
                  <div className="carrier-table__col carrier-table__col--carrier">Airline</div>
                  <div
                    className="carrier-table__col carrier-table__col--metric carrier-table__col--sortable"
                    onClick={() => handleSort('on_time_pct')}
                  >
                    On-Time % <SortIcon field="on_time_pct" />
                  </div>
                  <div className="carrier-table__col carrier-table__col--bar" style={{ position: 'relative' }}>
                    Performance
                    <span
                      className="carrier-table__info-icon"
                      onMouseEnter={() => setShowTooltip(true)}
                      onMouseLeave={() => setShowTooltip(false)}
                    >
                      i
                    </span>
                    {showTooltip && (
                      <div className="carrier-table__tooltip">
                        Bar length represents the absolute on-time arrival percentage (flights arriving within 15 minutes of schedule)
                      </div>
                    )}
                  </div>
                  <div
                    className="carrier-table__col carrier-table__col--metric carrier-table__col--sortable"
                    onClick={() => handleSort('avg_delay')}
                  >
                    Avg Delay <SortIcon field="avg_delay" />
                  </div>
                  <div
                    className="carrier-table__col carrier-table__col--metric carrier-table__col--sortable"
                    onClick={() => handleSort('n_flights')}
                  >
                    Flights <SortIcon field="n_flights" />
                  </div>
                </div>

                <div className="carrier-table__body">
                  {sortedData.map((carrier, idx) => {
                    const config = CARRIER_CONFIG[carrier.code] || { name: carrier.code, color: '#6b7280' }
                    const perfColor = getPerformanceColor(carrier.on_time_pct)

                    return (
                      <div
                        key={carrier.code}
                        className="carrier-table__row"
                        style={{ animationDelay: `${idx * 40}ms` }}
                      >
                        <div className="carrier-table__col carrier-table__col--rank">
                          <span
                            className="carrier-table__rank-badge"
                            style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
                          >
                            {idx + 1}
                          </span>
                        </div>
                        <div className="carrier-table__col carrier-table__col--carrier">
                          <span
                            className="carrier-table__carrier-dot"
                            style={{ backgroundColor: config.color }}
                          />
                          <span className="carrier-table__carrier-name">{config.name}</span>
                          <span className="carrier-table__carrier-code">{carrier.code}</span>
                        </div>
                        <div className="carrier-table__col carrier-table__col--metric">
                          <span className="carrier-table__value" style={{ color: perfColor }}>
                            {(carrier.on_time_pct * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="carrier-table__col carrier-table__col--bar">
                          <div className="carrier-table__bar">
                            <div
                              className="carrier-table__bar-fill"
                              style={{
                                width: `${carrier.on_time_pct * 100}%`,
                                backgroundColor: perfColor
                              }}
                            />
                          </div>
                        </div>
                        <div className="carrier-table__col carrier-table__col--metric">
                          <span className="carrier-table__value">
                            {carrier.avg_delay.toFixed(1)} min
                          </span>
                        </div>
                        <div className="carrier-table__col carrier-table__col--metric">
                          <span className="carrier-table__value carrier-table__value--muted">
                            {carrier.n_flights >= 1000000
                              ? `${(carrier.n_flights / 1000000).toFixed(1)}M`
                              : `${(carrier.n_flights / 1000).toFixed(0)}K`}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="carrier-layout__sidebar">
            <div className="finding-card">
              <h4>Top Performers</h4>
              <p>Delta Air Lines and Hawaiian Airlines rank highest in on-time arrival performance, each exceeding 85% on-time, with the lowest average delays among major carriers.</p>
            </div>
            <div className="finding-card">
              <h4>Operational Scale</h4>
              <p>Southwest Airlines operates the highest flight volume (8.0M flights) while maintaining competitive on-time performance (81.3%), illustrating the trade-off between scale and schedule reliability.</p>
            </div>
            <div className="finding-card">
              <h4>Budget Trade-offs</h4>
              <p>Low-cost carriers trail premium airlines by up to 11.5 percentage points in on-time performance, reflecting differences in scheduling buffers, fleet utilization, and operational slack.</p>
            </div>
            <div className="finding-card">
              <h4>Delay Gap</h4>
              <p>Bottom-ranked carriers average 7.1x longer delays than top performers (12.1 minutes vs. 1.7 minutes), reflecting large reliability gaps.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default CarrierPerformance
