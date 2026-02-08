import { useEffect, useState } from 'react'
import Hero from '../components/Hero'
import ForecastChart from '../components/ForecastChart'
import ModelComparison from '../components/ModelComparison'
import ErrorAnalysis from '../components/ErrorAnalysis'
import FeatureImportance from '../components/FeatureImportance'
import WeatherImpactChart from '../components/WeatherImpactChart'
import CarrierPerformance from '../components/CarrierPerformance'
import Methods from '../components/Methods'
import References from '../components/References'
import Footer from '../components/Footer'
import { assetUrl } from '../utils/helpers.js'

function Forecasting() {
  const [forecastData, setForecastData] = useState(null)
  const [forecastLoading, setForecastLoading] = useState(true)
  const [forecastError, setForecastError] = useState(null)

  useEffect(() => {
    const loadData = async () => {
      try {
        setForecastLoading(true)
        const response = await fetch(assetUrl('data/delay_forecasts.json'))
        if (!response.ok) throw new Error('Failed to load forecast data')
        const data = await response.json()
        setForecastData(data)
      } catch (err) {
        setForecastError('Failed to load forecast data')
      } finally {
        setForecastLoading(false)
      }
    }
    loadData()
  }, [])

  return (
    <>
      <Hero
        kicker="Route Delay Forecasting"
        title="Forecasting U.S. Route-Level Flight Delays"
        subtitle={<>
          <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
            Flight delays rarely happen in isolation. On any given day, weather, congestion, and network pressure combine to create predictable patterns of disruption across specific routes. When operations teams know in advance that a route is likely to experience delays, they can proactively rebook passengers, adjust crew assignments, and plan gate usage more effectively.
          </span>
          <span style={{ display: 'block' }}>
            This project explores whether machine learning can forecast tomorrow's average arrival delay for a given route, providing airlines and airports with early warning signals to manage downstream disruptions.
          </span>
        </>}
      />

      <div className="data-source-bar">
        <div className="container">
          <div className="data-source-box">
            <div className="data-source-item">
              <span className="data-source-label">Flight Data</span>
              <span className="data-source-value">BTS On-Time Performance</span>
            </div>
            <div className="data-source-divider" />
            <div className="data-source-item">
              <span className="data-source-label">Weather</span>
              <span className="data-source-value">Open-Meteo Historical API</span>
            </div>
            <div className="data-source-divider" />
            <div className="data-source-item">
              <span className="data-source-label">Time Span</span>
              <span className="data-source-value">January 2019 – June 2025</span>
            </div>
            <div className="data-source-divider" />
            <div className="data-source-item">
              <span className="data-source-label">Coverage</span>
              <span className="data-source-value">50 Busiest U.S. Routes</span>
            </div>
            <div className="data-source-divider" />
            <div className="data-source-item">
              <span className="data-source-label">Models</span>
              <span className="data-source-value">XGBoost, LightGBM, LSTM, TCN</span>
            </div>
          </div>
        </div>
      </div>

      <WeatherImpactChart />
      <CarrierPerformance />
      <ForecastChart forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <ModelComparison forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <ErrorAnalysis forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <FeatureImportance />
      <Methods />
      <References />
      <Footer />
    </>
  )
}

export default Forecasting
