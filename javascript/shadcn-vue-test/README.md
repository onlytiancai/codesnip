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

