// toggle between outbound/inbound for a city pair
function DirectionToggle({ cityPair, direction, onChange, availableRoutes }) {
  if (!cityPair) return null

  const [city1, city2] = cityPair.split('-')
  const outboundRoute = `${city1}-${city2}`
  const inboundRoute = `${city2}-${city1}`

  const outboundAvailable = availableRoutes?.includes(outboundRoute)
  const inboundAvailable = availableRoutes?.includes(inboundRoute)

  return (
    <div className="direction-toggle">
      <button
        className={`direction-btn ${direction === 'outbound' ? 'direction-btn--active' : ''}`}
        onClick={() => onChange('outbound')}
        disabled={!outboundAvailable}
        title={outboundAvailable ? outboundRoute : `${outboundRoute} (not available)`}
      >
        {city1} &rarr; {city2}
      </button>
      <button
        className={`direction-btn ${direction === 'inbound' ? 'direction-btn--active' : ''}`}
        onClick={() => onChange('inbound')}
        disabled={!inboundAvailable}
        title={inboundAvailable ? inboundRoute : `${inboundRoute} (not available)`}
      >
        {city2} &rarr; {city1}
      </button>
    </div>
  )
}

export default DirectionToggle
