function Methods() {
  return (
    <section id="methods" className="section section--alt">
      <div className="container" data-aos="fade-up">
        <div className="methods-section">
          <h2>Methodology</h2>

          <dl className="methods-list">
            <div className="methods-list__item">
              <dt>Data Processing</dt>
              <dd>
                Training data comes from the BTS On-Time Performance database<sup><a href="#ref-4">4</a></sup>, processed into daily arrival delay
                averages per route. Models are trained on 50 routes to increase data diversity, with the top 20 by
                traffic volume displayed in the dashboard. Features include temporal patterns, lag values (1-28 days),
                rolling statistics, route characteristics, and weather features (including hourly-derived operating-hour
                aggregates). Days with zero flights have delay values forward-filled up to 7 days to avoid gaps in lag
                calculations. A chronological split divides the data into training (Jan 2019 - Dec 2023), validation
                (Jan 2024 - Jun 2024), and test (Jul 2024 - Jun 2025) periods to prevent future data from leaking into
                training. Route-level aggregate statistics used for imputation and encoding are computed exclusively from
                the training period.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Forecasting Models</dt>
              <dd>
                All four models are trained as single global regressors on all 50 routes pooled together,
                with route identity captured through encoded features rather than building separate per-route models.
                We compare baselines (naive lag-1, 7-day moving average), gradient boosting (XGBoost, LightGBM),
                and deep learning (LSTM with attention, TCN). The central question is whether deep learning
                provides any advantage over gradient boosting for this type of tabular time series, or if
                well-engineered features are sufficient on their own.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Weather Integration</dt>
              <dd>
                Weather conditions at origin and destination airports come from Open-Meteo's ERA5
                reanalysis<sup><a href="#ref-3">3</a></sup>. Hourly data is aggregated into operating-hour
                features (peak wind, precipitation, storm-hour counts, morning/evening severity scores)
                that complement daily summaries. This lets models distinguish between all-day drizzle
                and a brief thunderstorm during peak departure hours.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Feature Design</dt>
              <dd>
                Gradient boosting models (XGBoost, LightGBM) use the full feature set including explicit lags,
                rolling statistics, route metadata, and all weather features. The sequence models (LSTM, TCN) use a
                curated subset focused on the raw delay signal, calendar, weather, and hourly-derived severity features
                in 28-day sliding windows. Pre-computed lag and rolling features conflicted with the temporal patterns
                the networks learn from the input window, and constant route-level features added noise without useful
                variation across timesteps. LSTM and TCN targets are standardized to prevent MSE loss from being
                dominated by high-delay outliers.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Hyperparameter Tuning</dt>
              <dd>
                All four model architectures are tuned with Optuna<sup><a href="#ref-5">5</a></sup> Bayesian optimization (50 trials each) using
                validation MAE as the objective. Search spaces include learning rate, regularization strength,
                tree depth / network width, and dropout. Neural network trials use MedianPruner for early
                termination of underperforming configurations. Tuned parameters are saved as JSON and loaded
                automatically by the training scripts.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Validation Strategy</dt>
              <dd>
                In addition to the primary held-out test period, a 4-fold walk-forward validation is performed
                with expanding training windows (6-month test periods from Jan 2023 through Dec 2024). This
                provides mean and standard deviation estimates for each metric across different time windows,
                giving confidence that reported performance is not an artifact of a single favorable test period.
                The final production models are trained on all data through Dec 2023 and evaluated on the
                Jul 2024 - Jun 2025 test set.
              </dd>
            </div>
          </dl>
        </div>

        <div className="methods-section" style={{ marginTop: 'var(--space-3xl)' }}>
          <h2>Limitations</h2>

          <dl className="methods-list">
            <div className="methods-list__item">
              <dt>Daily Aggregation</dt>
              <dd>
                Delays are averaged per route per day, which masks flight-level variation and extreme outliers
                within a single day. A route showing 10 minutes average delay could contain flights ranging
                from 60 minutes early to 3 hours late. Individual flight predictions would require a different
                modeling approach.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Network Effects</dt>
              <dd>
                Each route is modeled independently. Delay propagation across the airline network, where an
                upstream delay at a hub cascades to connecting flights, is not captured. A flight departing
                Atlanta for Miami does not know that the inbound aircraft was delayed in Chicago.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Route Coverage</dt>
              <dd>
                Models are trained on 50 high-volume directional routes, with the top 20 displayed in the
                dashboard. These represent a small fraction of all U.S. domestic corridors. Performance on
                lower-volume routes, regional corridors, or routes with different carrier mixes may differ
                substantially from what is shown here.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Carrier Aggregation</dt>
              <dd>
                Each route aggregates flights across all carriers operating that corridor. A LAX-JFK forecast
                reflects the combined performance of Delta, American, United, and JetBlue on that route. As shown
                in the Carrier Performance section, airlines vary significantly in on-time rates, but these
                carrier-specific effects are averaged out in our route-level predictions. A model that includes
                carrier identity as a feature could capture this variation.
              </dd>
            </div>
            <div className="methods-list__item">
              <dt>Extreme Delay Prediction</dt>
              <dd>
                All models struggle to predict delays at their extremes. While they effectively identify when a
                larger-than-typical delay is upcoming, they consistently underpredict the severity of extreme
                events. This is inherent to regression models trained on skewed distributions: optimizing for
                average loss pulls predictions toward central tendencies, extreme delays are rare in training
                data so tail patterns are learned less robustly, and tree-based models cannot extrapolate beyond
                values seen in their leaf nodes.
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  )
}

export default Methods
