import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navigation from './components/Navigation'
import Forecasting from './pages/Forecasting'
import Live from './pages/Live'
import Monitoring from './pages/Monitoring'

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
        <Route path="/" element={<Live />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/study" element={<Forecasting />} />
        <Route path="/forecasting" element={<Navigate to="/study" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
