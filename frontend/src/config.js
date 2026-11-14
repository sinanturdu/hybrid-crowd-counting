// NOTE: Vite environment variables prefixed with VITE_ are embedded into the
// client-side JavaScript bundle at build time and are NOT secret.
// Always restrict VITE_GOOGLE_MAPS_API_KEY in Google Cloud Console (HTTP referrers + Maps JavaScript API only).
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export const GOOGLE_MAPS_API_KEY =
  import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";

export const GOOGLE_MAPS_MAP_ID =
  import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || "DEMO_MAP_ID";
