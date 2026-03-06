export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },
  modules: [
    ["@nuxt/ui", { fonts: false }],
    "@nuxt/icon"
  ],
  css: ["~/assets/css/main.css"]
});