import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API is the FastAPI service, not this dev server. Proxying rather than
// hardcoding an origin keeps the browser on one origin, so there is no CORS
// configuration to get wrong and no second URL to keep in step.
export default defineConfig({
  plugins: [react()],
  // Plotly is one 4.8 MB chunk, split out and loaded only when an answer has
  // a chart. It cannot usefully be split further, so the warning is raised
  // past it rather than left to be ignored on every build.
  build: { chunkSizeWarningLimit: 5000 },
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
