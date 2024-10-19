import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 5173, // Make sure this port is defined
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // Backend server
        changeOrigin: true,
      },
    },

  },
})
