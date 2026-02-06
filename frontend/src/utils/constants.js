// static configuration, lookup tables, and color palettes
export const MONTH_NAMES = [
  '',
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

// Carrier name aliases for cleaner display
export const CARRIER_ALIASES = {
  'ExpressJet Airlines LLC d/b/a aha!': 'ExpressJet (aha!)',
};

export const MODEL_COLORS = {
  XGBoost: '#1e40af',
  LightGBM: '#513288',
  LSTM: '#842362',
  TCN: '#c8102e',
  'Moving Average': '#9ca3af',
  Naive: '#6b7280',
}

// Colorblind-friendly palette with 22 distinct colors (cool/neutral, on-theme)
export const MARKET_COLORS = [
  '#1F4B99', '#4E6FB8', '#7896CE', '#AFC2E6', '#C7D6ED', // blues
  '#0F6A6A', '#2F8F8F', '#5BA9A9', '#8CC6C6', '#B7E0DF', // teals
  '#3A6E3A', '#5F915F', '#8AB78A', '#B7D8B7', '#D9EBD9', // greens
  '#6B5FA5', '#8A7BC1', '#A897DD', '#C3B3F0', '#DDD3FF', // violets
  '#586F7C', '#7A8E9A' // neutrals
];
