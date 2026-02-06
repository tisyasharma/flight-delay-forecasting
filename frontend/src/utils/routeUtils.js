// deduplicate directional routes into sorted city pairs (LAX-SFO and SFO-LAX -> LAX-SFO)
export function extractCityPairs(routes) {
  const pairSet = new Set()
  routes.forEach(route => {
    const [origin, dest] = route.split('-')
    const sortedPair = [origin, dest].sort().join('-')
    pairSet.add(sortedPair)
  })
  return Array.from(pairSet).sort()
}

// build directional route string from city pair + direction
export function buildDirectionalRoute(cityPair, direction) {
  const [city1, city2] = cityPair.split('-')
  return direction === 'outbound' ? `${city1}-${city2}` : `${city2}-${city1}`
}

// display format with bidirectional arrow
export function formatCityPairDisplay(cityPair) {
  const [city1, city2] = cityPair.split('-')
  return `${city1} \u2194 ${city2}`
}

// get both directions for a city pair
export function getDirectionalRoutes(cityPair) {
  const [city1, city2] = cityPair.split('-')
  return {
    outbound: `${city1}-${city2}`,
    inbound: `${city2}-${city1}`
  }
}
