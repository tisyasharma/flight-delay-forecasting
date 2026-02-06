import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navigation from './components/Navigation'
import Forecasting from './pages/Forecasting'

function App() {
  useEffect(() => {
    if (window.AOS) {
      window.AOS.init({ once: true, duration: 600, easing: 'ease-out' })
    }
  }, [])

  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Navigation />
      <Routes>
        <Route path="/" element={<Navigate to="/forecasting" replace />} />
        <Route path="/forecasting" element={<Forecasting />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
