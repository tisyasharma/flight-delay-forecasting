import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/flight-delay-forecasting/' : '/',
  plugins: [react()],
  server: {
    port: 8000
  }
}))
