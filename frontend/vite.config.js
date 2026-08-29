export default {
  // TWO entry points: `/` is the landing page and `/app` is the workspace.
  // Without listing both here Vite builds only index.html, and app.html never
  // reaches dist/ even though the production rewrite points at it.
  //
  // In DEV the rewrite does not exist (it lives in vercel.json), so the
  // workspace is at /app.html.
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        app: "app.html",
      },
      output: {
        // Keep the workspace shell's two static warm-up targets stable in
        // dist; the landing page cannot discover Vite's hashed names later.
        entryFileNames: (chunk) =>
          chunk.name === "app" ? "app.js" : "assets/[name]-[hash].js",
        assetFileNames: (asset) =>
          asset.name === "app.css" ? "styles.css" : "assets/[name]-[hash][extname]",
      },
    },
  },
  server: {
    // Honour the PORT the host environment assigns (preview tooling sets it);
    // 5173 stays the default for a bare `npm run dev`.
    port: Number(process.env.PORT) || 5173,
    allowedHosts: [".mrchloep.com"],
    // This workspace can exhaust its inotify watcher limit; poll instead so
    // the dev server doesn't crash with ENOSPC. Hot reload still works.
    watch: {
      usePolling: true,
      interval: 1000,
      binaryInterval: 3000,
    },
    proxy: {
      // Backend routes already start with /api — no rewrite needed.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
};
