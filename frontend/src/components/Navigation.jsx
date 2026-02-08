import { useState, useEffect } from 'react'

const NAV_SECTIONS = [
  { id: 'weather-impact', label: 'Weather' },
  { id: 'carrier-performance', label: 'Airlines' },
  { id: 'forecast', label: 'Forecasts' },
  { id: 'model-comparison', label: 'Models' },
  { id: 'error-analysis', label: 'Error Analysis' },
  { id: 'feature-importance', label: 'Key Drivers' },
  { id: 'conclusion', label: 'Findings' },
  { id: 'methods', label: 'Methods' },
]

function Navigation() {
  const [activeSection, setActiveSection] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)

        if (visible.length > 0) {
          setActiveSection(visible[0].target.id)
        }
      },
      { rootMargin: '-80px 0px -60% 0px', threshold: 0 }
    )

    NAV_SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [])

  const handleClick = (e, id) => {
    e.preventDefault()
    setMenuOpen(false)
    const el = document.getElementById(id)
    if (el) {
      const navHeight = 56
      const top = el.getBoundingClientRect().top + window.scrollY - navHeight
      window.scrollTo({ top, behavior: 'smooth' })
    }
  }

  return (
    <nav className="nav">
      <div className="nav__inner">
        <a
          href="#"
          className="brand"
          onClick={(e) => {
            e.preventDefault()
            setMenuOpen(false)
            window.scrollTo({ top: 0, behavior: 'smooth' })
          }}
        >
          Route Delay Forecasting
        </a>
        <button
          className={`nav__toggle ${menuOpen ? 'nav__toggle--open' : ''}`}
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle navigation menu"
          aria-expanded={menuOpen}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>
        <div className={`nav__links ${menuOpen ? 'nav__links--open' : ''}`}>
          {NAV_SECTIONS.map(({ id, label }) => (
            <a
              key={id}
              href={`#${id}`}
              className={activeSection === id ? 'nav__link--active' : ''}
              onClick={(e) => handleClick(e, id)}
            >
              {label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navigation
