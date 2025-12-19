# Vue 3 + TypeScript + Vite

## setup

    pnpm create vite shadcn-vue-test
    ◇  Select a framework:
    │  Vue
    │
    ◇  Select a variant:
    │  TypeScript

    cd shadcn-vue-test
    pnpm install
    pnpm dev

    pnpm add -D tailwindcss postcss autoprefixer

    在 VS Code 扩展里搜索并安装：

    👉 Tailwind CSS IntelliSense（官方）

postcss.config.js

    export default {
    plugins: {
        tailwindcss: {},
        autoprefixer: {},
    },
    }

tailwind.config.js

    /** @type {import('tailwindcss').Config} */
    export default {
    darkMode: ["class"],
    content: [
        "./index.html",
        "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        container: {
        center: true,
        padding: "2rem",
        screens: {
            "2xl": "1400px",
        },
        },
        extend: {},
    },
    plugins: [],
    }


src/style.css

    @tailwind base;
    @tailwind components;
    @tailwind utilities;

shadcn-vue

    pnpm add -D shadcn-vue

    git diff vite.config.ts 
    @@ -1,7 +1,13 @@
    +import { fileURLToPath, URL } from 'node:url'
    import { defineConfig } from 'vite'
    import vue from '@vitejs/plugin-vue'
    
    // https://vite.dev/config/
    export default defineConfig({
    plugins: [vue()],
    +  resolve: {
    +    alias: {
    +      '@': fileURLToPath(new URL('./src', import.meta.url)),
    +    },
    +  },
    })

    git diff tsconfig.json
    @@ -3,5 +3,13 @@
    "references": [
        { "path": "./tsconfig.app.json" },
        { "path": "./tsconfig.node.json" }
    -  ]
    +  ],
    +  "compilerOptions": {
    +    "baseUrl": ".",
    +    "paths": {
    +      "@/*": [
    +        "src/*"
    +      ]
    +    }
    +  }
    }

    git diff tsconfig.app.json 
    @@ -10,7 +10,13 @@
        "noUnusedParameters": true,
        "erasableSyntaxOnly": true,
        "noFallthroughCasesInSwitch": true,
    -    "noUncheckedSideEffectImports": true
    +    "noUncheckedSideEffectImports": true,
    +    "baseUrl": ".",
    +    "paths": {
    +      "@/*": [
    +        "src/*"
    +      ]
    +    }    
    },
    "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"]
    }

    pnpm shadcn-vue init

add button

    pnpm shadcn-vue add button
    pnpm add -D @tailwindcss/postcss
    pnpm add -D tw-animate-css



