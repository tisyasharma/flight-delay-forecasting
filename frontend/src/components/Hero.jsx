function Hero({ kicker, title, subtitle }) {
  const defaultSubtitle = (
    <>
      <span style={{ display: 'block', marginBottom: 'var(--space-md)' }}>
        Flight delays rarely happen in isolation. On any given day, weather, congestion, and network pressure combine to create predictable patterns of disruption across specific routes. When operations teams know in advance that a route is likely to experience delays, they can proactively rebook passengers, adjust crew assignments, and plan gate usage more effectively.
      </span>
      <span style={{ display: 'block' }}>
        This project explores whether machine learning can forecast tomorrow's average arrival delay for a given route, providing airlines and airports with early warning signals to manage downstream disruptions.
      </span>
    </>
  )

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
          {subtitle || defaultSubtitle}
        </p>
      </div>
    </header>
  )
}

export default Hero
