import { useState } from 'react'

function Methods() {
  const [methodsExpanded, setMethodsExpanded] = useState(false)
  const [limitationsExpanded, setLimitationsExpanded] = useState(false)

  const keyFindings = [
    {
      number: '1',
      content: <>Gradient boosting outperformed deep learning in this study. On a dataset of 50 high-volume U.S. routes with daily aggregated delays, XGBoost and LightGBM achieved lower error than LSTM and TCN architectures on every route, with a 77.7% hit rate. For this tabular time-series task with well-engineered features, deep learning offered no advantage. However, recent research suggests that hybrid architectures combining CNNs, LSTMs, and graph neural networks can outperform gradient boosting by capturing spatio-temporal dependencies across airport networks.<sup><a href="#ref-3">3</a></sup></>
    },
    {
      number: '2',
      content: 'Feature engineering mattered more than model architecture. In ablation testing, adding weather features reduced XGBoost MAE by 10.3%. Notably, gradient boosting without weather data still outperformed deep learning trained on the full feature set. Within each model family, performance differences were negligible, suggesting similar algorithmic ceilings rather than tuning limitations.'
    },
    {
      number: '3',
      content: 'Forecast difficulty varied widely across routes and seasons. MAE ranged from 3.8 minutes on HNL-OGG to 15.8 minutes on FLL-ATL, a fourfold spread. Hawaii inter-island routes with stable weather were the most predictable, while Northeast corridors and Atlanta hub connections were the most challenging. Spring was consistently the hardest season, while fall produced the lowest errors.'
    },
    {
      number: '4',
      content: 'Gradient boosting provides practical value for high-volume routes. A model combining lagged delay history and weather features produced reliable next-day forecasts that could support operational decisions such as gate assignment and crew scheduling. Real-world deployment would require broader route coverage and live data validation.'
    }
  ]

  const methodologyItems = [
    {
      title: 'Data Processing',
      content: <>Training data is sourced from the Bureau of Transportation Statistics On-Time Performance database<sup><a href="#ref-2">2</a></sup> and processed into daily average arrival delays per route. Models are trained on 50 high-volume directional routes to increase data diversity, with the top 20 routes by traffic volume displayed in the dashboard. Features include calendar variables, lagged delays from 1 to 28 days, rolling statistics, route-level aggregates, and weather features derived from hourly observations.</>
    },
    {
      title: 'Data Continuity',
      content: 'Days with zero flights are forward-filled up to seven days to preserve continuity in lag calculations. A strictly chronological split divides the data into training (January 2019 through December 2023), validation (January through June 2024), and test (July 2024 through June 2025) periods to prevent future data leakage. All aggregate statistics used for encoding and imputation are computed exclusively from the training period.'
    },
    {
      title: 'Forecasting Models',
      content: 'All models are trained as single global regressors on pooled route data, with route identity encoded as features rather than training separate per-route models. Baselines include naive lag-1 and seven-day moving average forecasts. We compare gradient boosting models (XGBoost and LightGBM) against deep learning approaches (LSTM with attention and TCN) to assess whether neural architectures provide an advantage over feature-based methods in this setting.'
    },
    {
      title: 'Weather Integration',
      content: <>Weather data is obtained from Open-Meteo's ERA5 reanalysis.<sup><a href="#ref-1">1</a></sup> Hourly conditions at origin and destination airports are aggregated into operating-hour features such as peak wind, precipitation totals, storm-hour counts, and morning and evening severity scores. This design allows models to distinguish brief high-impact events from persistent mild conditions.</>
    },
    {
      title: 'Feature Design',
      content: 'Gradient boosting models use the full engineered feature set. Sequence models use a curated subset focused on raw delay history, calendar features, and weather signals in 28-day sliding windows. Pre-computed lag and rolling statistics were excluded from neural networks to avoid redundancy with learned temporal representations, and constant route-level features were removed to reduce noise. Targets for LSTM and TCN models are standardized to prevent extreme delays from dominating the loss function.'
    },
    {
      title: 'Hyperparameter Tuning',
      content: <>All models are tuned using Optuna<sup><a href="#ref-4">4</a></sup> Bayesian optimization with validation MAE as the objective. Each architecture undergoes 50 trials. Neural network trials use early stopping via MedianPruner. Final tuned parameters are saved and reused in training scripts.</>
    },
    {
      title: 'Validation Strategy',
      content: 'In addition to evaluation on the held-out test period, a four-fold walk-forward validation is performed using expanding training windows and six-month test periods from January 2023 through December 2024. This provides confidence that reported performance reflects consistent behavior over time rather than a favorable single split.'
    }
  ]

  const limitationItems = [
    {
      title: 'Daily Aggregation',
      content: 'Delays are averaged per route per day, which masks flight-level variability and extreme outliers within a given day. Predicting individual flight delays would require a different modeling approach.'
    },
    {
      title: 'Network Effects',
      content: 'Routes are modeled independently, so network effects such as delay propagation through hubs are not captured. An outbound flight does not incorporate information about delays on inbound aircraft.'
    },
    {
      title: 'Route Coverage',
      content: 'The analysis focuses on 50 high-volume routes, representing a small fraction of U.S. domestic corridors. Performance on lower-volume or regional routes may differ substantially.'
    },
    {
      title: 'Carrier Aggregation',
      content: 'Each route aggregates flights across all operating carriers. Airline-specific operational differences are averaged out, even though carriers vary significantly in on-time performance. Including carrier identity could capture this variation.'
    },
    {
      title: 'Extreme Delay Prediction',
      content: 'All models struggle to predict extreme delay events. While they detect periods of elevated delay risk, they underpredict the magnitude of the most severe disruptions due to skewed delay distributions and the rarity of tail events in training data.'
    }
  ]

  const INITIAL_SHOW = 3

  const visibleMethods = methodsExpanded ? methodologyItems : methodologyItems.slice(0, INITIAL_SHOW)
  const visibleLimitations = limitationsExpanded ? limitationItems : limitationItems.slice(0, INITIAL_SHOW)

  const hiddenMethodsCount = methodologyItems.length - INITIAL_SHOW
  const hiddenLimitationsCount = limitationItems.length - INITIAL_SHOW

  return (
    <section id="methods" className="section section--alt">
      <div className="container" data-aos="fade-up">
        <div className="methods-section">
          <h2>Results Overview</h2>

          <p style={{ marginBottom: 'var(--space-md)' }}>
            This project evaluates whether modern machine learning models can provide actionable next-day delay forecasts for high-volume U.S. airline routes using daily aggregated data.
          </p>

          <p style={{ marginBottom: 'var(--space-md)' }}>
            Across 50 major domestic routes, gradient boosting models consistently outperformed deep learning. XGBoost and LightGBM achieved lower error than LSTM and TCN models on every route, with a 77.7% hit rate, defined as predictions falling within ±15 minutes of the actual average delay.
          </p>

          <p style={{ marginBottom: 'var(--space-md)' }}>
            Forecast difficulty varied substantially across corridors. Average error ranged from 3.8 minutes on HNL-OGG to 15.8 minutes on FLL-ATL, reflecting differences in weather volatility, congestion, and operational complexity. Seasonal effects were pronounced: spring consistently produced the highest forecast errors, while fall yielded the lowest, indicating meaningful seasonal variation in predictability across routes.
          </p>

          <p style={{ marginBottom: 'var(--space-lg)' }}>
            For the routes studied, a gradient boosting model incorporating lagged delay history and weather features produced forecasts that are operationally useful for planning tasks such as gate assignment and crew scheduling. Broader deployment would require validation on additional routes and integration with live operational data.
          </p>
        </div>

        <div className="methods-section" style={{ marginTop: 'var(--space-3xl)' }}>
          <h2>Key Findings</h2>

          <ol className="key-findings-list">
            {keyFindings.map((item) => (
              <li key={item.number} className="key-findings-list__item">
                {item.content}
              </li>
            ))}
          </ol>
        </div>

        <div className="methods-section" style={{ marginTop: 'var(--space-3xl)' }}>
          <h2>Methodology</h2>

          <dl className="methods-list">
            {visibleMethods.map((item, index) => (
              <div
                key={item.title}
                className="methods-list__item"
                style={{
                  animation: methodsExpanded && index >= INITIAL_SHOW
                    ? `methodFadeIn 0.3s ease ${(index - INITIAL_SHOW) * 0.05}s forwards`
                    : 'none',
                  opacity: methodsExpanded && index >= INITIAL_SHOW ? 0 : 1
                }}
              >
                <dt>{item.title}</dt>
                <dd>{item.content}</dd>
              </div>
            ))}
          </dl>

          {hiddenMethodsCount > 0 && (
            <div style={{ borderTop: '1px solid var(--border)', marginTop: 'var(--space-md)', paddingTop: 'var(--space-sm)' }}>
              <button
                onClick={() => setMethodsExpanded(!methodsExpanded)}
                aria-expanded={methodsExpanded}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '4px 0',
                  color: 'var(--text-muted)',
                  fontSize: '13px',
                  fontWeight: '500'
                }}
              >
                <span style={{ transform: methodsExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: '10px' }}>&#9654;</span>
                {methodsExpanded ? 'Hide methods' : `View ${hiddenMethodsCount} additional methods`}
              </button>
            </div>
          )}
        </div>

        <div className="methods-section" style={{ marginTop: 'var(--space-3xl)' }}>
          <h2>Limitations</h2>

          <dl className="methods-list">
            {visibleLimitations.map((item, index) => (
              <div
                key={item.title}
                className="methods-list__item"
                style={{
                  animation: limitationsExpanded && index >= INITIAL_SHOW
                    ? `methodFadeIn 0.3s ease ${(index - INITIAL_SHOW) * 0.05}s forwards`
                    : 'none',
                  opacity: limitationsExpanded && index >= INITIAL_SHOW ? 0 : 1
                }}
              >
                <dt>{item.title}</dt>
                <dd>{item.content}</dd>
              </div>
            ))}
          </dl>

          {hiddenLimitationsCount > 0 && (
            <div style={{ borderTop: '1px solid var(--border)', marginTop: 'var(--space-md)', paddingTop: 'var(--space-sm)' }}>
              <button
                onClick={() => setLimitationsExpanded(!limitationsExpanded)}
                aria-expanded={limitationsExpanded}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '4px 0',
                  color: 'var(--text-muted)',
                  fontSize: '13px',
                  fontWeight: '500'
                }}
              >
                <span style={{ transform: limitationsExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: '10px' }}>&#9654;</span>
                {limitationsExpanded ? 'Hide limitations' : `View ${hiddenLimitationsCount} additional limitations`}
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

export default Methods
