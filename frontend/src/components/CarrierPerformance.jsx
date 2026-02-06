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

  const topCarriers = [...(data?.carriers || [])].sort((a, b) => b.on_time_pct - a.on_time_pct).slice(0, 2)
  const volumeLeader = [...(data?.carriers || [])].sort((a, b) => b.n_flights - a.n_flights)[0]
  const byDelay = [...(data?.carriers || [])].sort((a, b) => a.avg_delay - b.avg_delay)
  const bestDelay = byDelay[0]
  const worstDelay = byDelay[byDelay.length - 1]
  const delayRatio = bestDelay ? (worstDelay.avg_delay / bestDelay.avg_delay).toFixed(1) : '?'

  const formatFlightCount = (n) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
    return `${(n / 1000).toFixed(0)}K`
  }

  const dateRange = data?.date_range
  const dateRangeText = dateRange
    ? `${new Date(dateRange.start).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })} to ${new Date(dateRange.end).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}`
    : 'January 2019 to June 2025'

  return (
    <section id="carrier-performance" className="section">
      <div className="container" data-aos="fade-up">
        <p className="kicker">Industry Context</p>
        <h2>Carrier Performance Rankings</h2>
        <p style={{ marginBottom: 'var(--space-lg)' }}>
          Carrier choice affects delay outcomes, though our forecasting models operate at the route level and average across all airlines on a corridor. This is a simplification: carrier-specific effects within a route are not captured. Across the top 10 major U.S. carriers from {dateRangeText}, premium airlines consistently outperform low-cost alternatives on on-time arrivals, average delays, and severe disruptions.<sup><a href="#ref-4">4</a></sup> Regional operators (SkyWest, Endeavor, Envoy, Republic, etc.) are excluded as they fly under major airlines.
        </p>

        <div className="carrier-layout">
          <div className="carrier-layout__sidebar">
            <div className="finding-card">
              <h4>Top Performers</h4>
              <p>
                {topCarriers.map(c => CARRIER_CONFIG[c.code]?.name || c.code).join(' and ')} rank highest in on-time performance ({(topCarriers[0]?.on_time_pct * 100).toFixed(0)}% and {(topCarriers[1]?.on_time_pct * 100).toFixed(0)}% respectively).
              </p>
            </div>
            <div className="finding-card">
              <h4>Volume Leader</h4>
              <p>{CARRIER_CONFIG[volumeLeader?.code]?.name || volumeLeader?.code} operates the most flights ({formatFlightCount(volumeLeader?.n_flights)}) while maintaining {(volumeLeader?.on_time_pct * 100).toFixed(0)}% on-time performance.</p>
            </div>
            <div className="finding-card">
              <h4>Budget Trade-offs</h4>
              <p>{(() => {
                const sorted = [...(data?.carriers || [])].sort((a, b) => b.on_time_pct - a.on_time_pct)
                const best = sorted[0]
                const worst = sorted[sorted.length - 1]
                if (!best || !worst) return ''
                const gap = ((best.on_time_pct - worst.on_time_pct) * 100).toFixed(1)
                return `Budget carriers trail premium airlines by up to ${gap} percentage points in on-time performance.`
              })()}</p>
            </div>
            <div className="finding-card">
              <h4>Delay Gap</h4>
              <p>Bottom performers average {delayRatio}x longer delays than top performers ({worstDelay?.avg_delay.toFixed(1)} min vs {bestDelay?.avg_delay.toFixed(1)} min).</p>
            </div>
          </div>

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
        </div>
      </div>
    </section>
  )
}

export default CarrierPerformance
