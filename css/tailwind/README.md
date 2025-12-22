tailwind 练习

项目初始化

    pnpm create vite .
    pnpm add tailwindcss @tailwindcss/vite

    vi vite.config.ts
    import { defineConfig } from 'vite'
    import tailwindcss from '@tailwindcss/vite'

    export default defineConfig({
    plugins: [
        tailwindcss(),
    ],
    })

    vi src/style.css
    @import "tailwindcss";

    pnpm dev 

    vi index.html
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="/src/style.css" rel="stylesheet">
    </head>
    <body>
    <h1 class="text-3xl font-bold underline">
        Hello world!
    </h1>
    </body>
    </html>    