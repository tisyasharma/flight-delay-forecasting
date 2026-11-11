/*
Open-Meteo credit that its CC BY 4.0 licence requires wherever its data appear:
the source, a licence link, and a note that the values shown are derived.
*/
function WeatherCredit() {
  return (
    <span className="weather-credit">
      Weather data by{' '}
      <a href="https://open-meteo.com/" target="_blank" rel="noopener noreferrer">Open-Meteo.com</a>, licensed under{' '}
      <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>.
      This site's weather values and forecasts are derived from it.
    </span>
  )
}

export default WeatherCredit
