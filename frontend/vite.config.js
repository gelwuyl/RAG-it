export default {
  // Two entry points, not one: `/` is the landing page and `/app` is the
  // workspace. Without this Vite only builds index.html and app.html never
  // reaches dist/, so the deploy 404s on /app with no build error to explain it.
  //
  // In DEV there is no /app rewrite (that lives in vercel.json), so the app is
  // at http://localhost:5173/app.html during development.
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        app: "app.html",
      },
    },
  },
  server: {
    port: 5173,
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
