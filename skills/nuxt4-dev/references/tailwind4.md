# Tailwind CSS v4 in Nuxt 4

Tailwind CSS v4 introduces significant changes from v3. This guide covers the key differences and how to use Tailwind v4 in Nuxt 4 projects.

## Installation

Tailwind v4 is installed as a direct dependency:

```bash
pnpm add tailwindcss
```

In `nuxt.config.ts`, the module is auto-detected in Nuxt 4, but you can explicitly configure it:

```typescript
export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },
  modules: [
    ["@nuxt/ui", { fonts: false }],  // Nuxt UI includes Tailwind
    "@nuxt/icon",
  ],
  css: ["~/assets/css/main.css"]
})
```

## CSS File Setup

Create `app/assets/css/main.css`:

```css
@import "tailwindcss";

/* Custom CSS variables */
:root {
  --color-primary: #3b82f6;
}

/* Custom utilities */
@utility text-balance {
  text-wrap: balance;
}
```

## Key Changes from v3 to v4

### 1. No More `@tailwind` Directives

**v3 (OLD):**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**v4 (NEW):**
```css
@import "tailwindcss";
```

### 2. CSS-First Configuration

Tailwind v4 uses CSS variables instead of `tailwind.config.js` for most configuration:

```css
@theme {
  --color-brand: #3b82f6;
  --font-display: "Inter", sans-serif;
  --breakpoint-3xl: 120rem;
}
```

### 3. Improved Performance

- Single CSS file import
- Faster build times
- Smaller output bundles
- No more PurgeCSS configuration needed

## Using with Nuxt UI

Nuxt UI v4 is built on top of Tailwind CSS v4. The integration is seamless:

```vue
<template>
  <div class="p-4">
    <UButton color="primary" variant="solid">
      Click me
    </UButton>

    <UCard class="mt-4">
      <template #header>
        <h3 class="text-lg font-semibold">Card Title</h3>
      </template>
      <p>Card content</p>
    </UCard>
  </div>
</template>
```

## Common Patterns

### Responsive Design

```vue
<template>
  <div class="
    grid
    grid-cols-1
    sm:grid-cols-2
    lg:grid-cols-3
    gap-4
  ">
    <!-- Content -->
  </div>
</template>
```

### Dark Mode

Nuxt UI handles dark mode automatically. For custom elements:

```vue
<template>
  <div class="bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
    Content
  </div>
</template>
```

### Custom Color Palettes

```css
@theme {
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-500: #3b82f6;
  --color-primary-600: #2563eb;
  --color-primary-700: #1d4ed8;
}
```

Then use in components:

```vue
<UButton color="primary-500" />
```

## Tips

1. **Use semantic class names** - Prefer `text-primary` over `text-blue-500`
2. **Leverage Nuxt UI** - Use `UButton`, `UCard` etc. instead of raw Tailwind when possible
3. **Check generated CSS** - Run `pnpm nuxt build` and inspect `.nuxt/` output
4. **CSS variables are your friend** - Use `--color-*` variables for theming
