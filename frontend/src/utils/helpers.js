// formatting and calculation helpers used across components

import { MONTH_NAMES, CARRIER_ALIASES } from './constants.js';

const fmt = new Intl.NumberFormat('en-US');

// format number with thousands separator, returns '-' for invalid
export function formatNumber(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '-';
  return fmt.format(Math.round(n));
}

// build a URL that respects Vite's configured base path
export function assetUrl(path) {
  const base = import.meta.env.BASE_URL || '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const normalizedPath = path.startsWith('/') ? path.slice(1) : path;
  return `${normalizedBase}${normalizedPath}`;
}

// format a 0-1 value as percentage string
export function formatPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  return (v * 100).toFixed(1) + '%';
}

// format airport code with optional city/state label
export function formatAirportLabel(code, city, state) {
  if (!code) return '';
  const place = city ? (state ? `${city}, ${state}` : city) : '';
  return place ? `${code} (${place})` : code;
}

// month number (1-12) to display name, 0 = "All months"
export function monthLabel(m) {
  if (!m) return 'All months';
  return MONTH_NAMES[m] || 'Month ' + m;
}

// year number to string, 0 = "All years"
export function yearLabel(y) {
  if (!y) return 'All years';
  return String(y);
}

// apply carrier display aliases
export function displayCarrier(name) {
  return CARRIER_ALIASES[name] || name;
}

// strip corporate suffixes for cleaner legend labels
export function legendLabel(name) {
  return displayCarrier(name)
    .replace(/\b(inc\.?|co\.?|corp\.?|corporation|l\.?l\.?c\.?)\b/gi, '')
    .replace(/[.,]+$/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

export const MODEL_DISPLAY_ORDER = ['xgboost', 'lightgbm', 'tcn', 'lstm', 'naive', 'ma']

// sort [key, model] entries by canonical display order
export function sortModelEntries(entries) {
  return [...entries].sort(([a], [b]) => {
    const ai = MODEL_DISPLAY_ORDER.indexOf(a)
    const bi = MODEL_DISPLAY_ORDER.indexOf(b)
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi)
  })
}

// simple debounce for resize handlers
export function debounce(fn, wait = 150) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}
