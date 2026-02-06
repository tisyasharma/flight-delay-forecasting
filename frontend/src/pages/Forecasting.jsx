import { useEffect, useState } from 'react'
import Hero from '../components/Hero'
import ForecastChart from '../components/ForecastChart'
import ModelComparison from '../components/ModelComparison'
import ErrorAnalysis from '../components/ErrorAnalysis'
import Conclusion from '../components/Conclusion'
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
        subtitle={<>A single delayed flight can ripple across an airline's network, affecting connecting passengers, crew schedules, and aircraft availability for hours.<sup><a href="#ref-1">1</a></sup> When operations teams know a delay is coming, they can rebook passengers earlier, move crew to cover later flights, and load the right amount of fuel instead of guessing. Airports benefit too, using arrival forecasts to assign gates more efficiently and keep ground crews ready.<sup><a href="#ref-2">2</a></sup> This project explores whether machine learning can predict tomorrow's average arrival delay for a given route, giving airlines and airports the advance warning they need to stay ahead of disruptions.</>}
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
      <FeatureImportance />
      <ForecastChart forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <ModelComparison forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <ErrorAnalysis forecastData={forecastData} loading={forecastLoading} error={forecastError} />
      <Conclusion />
      <Methods />
      <References />
      <Footer />
    </>
  )
}

export default Forecasting
