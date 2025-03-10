import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/upload': {
        target: 'http://localhost:5002',
        changeOrigin: true,
      },
      '/train': {
        target: 'http://localhost:5002',
        changeOrigin: true,
      },
      '/results': {
        target: 'http://localhost:5002',
        changeOrigin: true,
      },
      '/uploads': {
        target: 'http://localhost:5002',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:5002',
        changeOrigin: true,
      }
    }
  }
}) 