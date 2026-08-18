export default {
  // THREE entry points, not one: `/` is the landing page, `/app` is the
  // workspace, `/built` is the engineering write-up. Without listing each one
  // here Vite builds only index.html, the others never reach dist/, and the
  // deploy 404s on them with no build error to explain it.
  //
  // In DEV none of the rewrites exist (they live in vercel.json), so the pages
  // are at /app.html and /built.html during development.
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        app: "app.html",
        built: "built.html",
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
