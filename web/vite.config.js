import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API is the FastAPI service, not this dev server. Proxying rather than
// hardcoding an origin keeps the browser on one origin, so there is no CORS
// configuration to get wrong and no second URL to keep in step.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
