import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://devhyacinthe.github.io',
  base: 'myPortfolio',
  trailingSlash: 'always',
  integrations: [tailwind()],
});
