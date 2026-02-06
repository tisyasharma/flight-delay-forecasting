function Hero({ kicker, title, subtitle }) {
  return (
    <header id="top" className="hero">
      <div className="container">
        {kicker && (
          <p className="kicker" data-aos="fade-up">
            {kicker}
          </p>
        )}
        <h1 data-aos="fade-up">
          {title || 'Forecasting U.S. Route-Level Flight Delays'}
        </h1>
        <p className="lede" data-aos="fade-up">
          {subtitle || 'Comparing gradient boosting and deep learning approaches, tuned via Bayesian optimization and evaluated with walk-forward validation, to forecast daily arrival delays across 50 major U.S. routes using BTS flight records and Open-Meteo weather data (Jan 2019 to Jun 2025).'}
        </p>
      </div>
    </header>
  )
}

export default Hero
