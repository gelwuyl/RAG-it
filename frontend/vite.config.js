export default {
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
