// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to FastAPI so you never hit CORS issues in dev
    proxy: {
      "/predict": "http://localhost:8000",
      "/session":  "http://localhost:8000",
    },
  },
});
