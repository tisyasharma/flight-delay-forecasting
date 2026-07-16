import { useEffect, useState } from 'react'
import Hero from '../components/Hero'
import LiveForecastChart from '../components/LiveForecastChart'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'

function Live() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const response = await fetch(assetUrl('data/live_forecasts.json'))
        if (!response.ok) throw new Error('Failed to load live forecasts')
        setData(await response.json())
      } catch (err) {
        setError('Failed to load live forecasts')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  return (
    <>
      <Hero
        kicker="Live Forecasts"
        title="U.S. Route Delay Outlook"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            Daily seven-day forecasts of average arrival delay for the 50 busiest U.S. domestic
            routes, built from Bureau of Transportation Statistics flight records and Open-Meteo
            weather forecasts.
          </span>
          <span style={{ display: 'block' }}>
            Forecasts refresh every morning. The evaluation study behind the model
            (walk-forward validation, calibration, error analysis) is on the Study page.
          </span>
        </>}
      />
      <LiveForecastChart data={data} loading={loading} error={error} />
      <Footer />
    </>
  )
}

export default Live
