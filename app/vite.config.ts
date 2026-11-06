import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 1420 is fixed: tauri.conf.json devUrl and the engine's CORS list expect it.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: { target: "es2022", sourcemap: true },
});
