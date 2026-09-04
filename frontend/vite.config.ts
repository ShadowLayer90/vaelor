import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/v2/",
  plugins: [react()],
  build: {
    outDir: "../vaelor/www_v2",
    emptyOutDir: true,
    // No production source maps: this output is packaged verbatim into the
    // release wheel as `vaelor/www_v2/**` (pyproject package-data), so emitting
    // maps ships ~850 KB of dead weight and the full un-minified source in every
    // wheel. `tools/build_release.py` rebuilds the frontend and packages whatever
    // it emits, so the map suppression has to live here, not in a later strip
    // step. The dev server (`vite serve`) keeps its own maps regardless.
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/entry-[hash].js",
        // `[name]` restored so the split chunks are self-identifying — `vendor`
        // and the per-route chunks (`AgentCenter`, `Workloads`, …) each carry
        // their own name. Still content-hashed, so the stale-asset-recovery
        // shim in vaelor/frontend_routes.py keeps matching `<name>-<hash>.js`.
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/asset-[hash][extname]",
        // Keep the framework (react, react-dom, react/jsx-runtime and
        // react-dom's own scheduler runtime) in one stable `vendor` chunk so
        // that app-code changes reissue only the app chunks and the browser
        // keeps its cached framework. Everything else — including the
        // React.lazy route chunks in Overview.tsx — splits on its own.
        manualChunks(id) {
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) {
            return "vendor";
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:34001",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
