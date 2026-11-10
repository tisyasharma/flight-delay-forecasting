import { useEffect, useState } from 'react'

// 48 hours rather than 24 because scheduled runs can start hours after 09:30 UTC
const STALE_AFTER_HOURS = 48

// true once more than STALE_AFTER_HOURS separate the timestamp from nowMs
export function isStale(timestamp, nowMs) {
  return (nowMs - timestamp.getTime()) / 36e5 > STALE_AFTER_HOURS
}

/**
 * Current time in ms, re-read every minute and whenever the tab becomes visible, so date labels
 * and staleness checks catch up within a minute of a threshold, a clock change, or a wake.
 */
export function useMinuteClock() {
  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const refresh = () => setNowMs(Date.now())
    const interval = setInterval(refresh, 60 * 1000)
    document.addEventListener('visibilitychange', refresh)
    return () => {
      clearInterval(interval)
      document.removeEventListener('visibilitychange', refresh)
    }
  }, [])
  return nowMs
}
