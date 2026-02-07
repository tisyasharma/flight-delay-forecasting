function Conclusion() {
  return (
    <section id="conclusion" className="section">
      <div className="container" data-aos="fade-up">
        <div className="conclusion-content">
          <p className="kicker">Summary</p>
          <h2>Key Findings</h2>

          <ol className="findings-list">
            <li>
              <strong>Gradient boosting outperformed deep learning in this study.</strong> On our
              dataset of 50 high-volume U.S. routes with daily aggregated delays, XGBoost and
              LightGBM achieved lower error than LSTM and TCN on every route, with a 79.6% hit
              rate (predictions within 15 minutes of actual). For this tabular time-series task
              with well-engineered features, deep learning offered no advantage.
            </li>
            <li>
              <strong>Feature engineering mattered more than model architecture.</strong> In our
              ablation study, removing weather features increased XGBoost MAE by 10.3%. Notably,
              gradient boosting without weather data still outperformed deep learning with the
              full feature set. Within each model category, performance differences were negligible
              (XGBoost vs LightGBM both at 10.9 min MAE, LSTM vs TCN both at 13.1 min). Each model
              was tuned independently with Optuna, so this convergence likely reflects similar
              algorithmic foundations hitting similar ceilings rather than any methodological constraint.
            </li>
            <li>
              <strong>Forecast difficulty varied widely across routes.</strong> MAE ranged from
              3.8 minutes (HNL-OGG) to 15.1 minutes (FLL-ATL), a 4x spread. Hawaii inter-island
              routes with stable weather were easiest to predict, while Northeast corridors and
              Atlanta hub connections proved most challenging. Seasonal patterns also emerged:
              Fall had the lowest errors (8.8 min), Summer the highest (14.1 min).
            </li>
            <li>
              <strong>Practical utility for high-volume routes.</strong> For the routes studied,
              a gradient boosting model with lag features and weather data provided actionable
              next-day forecasts. The 79.6% hit rate suggests this approach could support
              operational decisions like gate assignments and crew scheduling, though real-world
              deployment would require validation on broader route coverage and live data.
            </li>
          </ol>
        </div>
      </div>
    </section>
  )
}

export default Conclusion
