import { useEffect, useState } from 'react'
import Hero from '../components/Hero'
import LiveForecastChart from '../components/LiveForecastChart'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'

const MODEL_CARD_URL =
  'https://github.com/tisyasharma/flight-delay-forecasting/blob/main/docs/model-card.md'

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
        kicker="Route Forecasts"
        title="U.S. Route Delay Outlook"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            Seven-day forecasts of average arrival delay for the 50 busiest U.S. domestic
            routes, built from Bureau of Transportation Statistics flight records and Open-Meteo
            weather forecasts.
          </span>
          <span style={{ display: 'block' }}>
            The chart shows the latest published run. A run that fails any of its checks, such as
            incomplete weather coverage or a horizon that would need recursion beyond the 90 days the
            backtest validates, publishes nothing, and the last published run stays up with its date
            marked. The{' '}
            <a href={MODEL_CARD_URL} target="_blank" rel="noopener noreferrer">model card</a>{' '}
            documents the evaluation behind this model, run on an earlier generation trained the same
            way: walk-forward validation, interval calibration, and the recursion-depth backtest. The
            Study page shows the February 2026 study that preceded it.
          </span>
        </>}
      />
      <LiveForecastChart data={data} loading={loading} error={error} />
      <Footer />
    </>
  )
}

export default Live
