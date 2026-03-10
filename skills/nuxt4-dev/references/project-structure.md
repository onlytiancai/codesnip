# Nuxt 4 Project Structure

Nuxt 4 uses the `app/` directory structure for better organization and flexibility. This guide covers the correct project layout and common pitfalls.

## Directory Structure

```
project/
├── app/                        # Main application directory (Nuxt 4)
│   ├── assets/
│   │   └── css/
│   │       └── main.css        # Global CSS (Tailwind imports)
│   ├── components/             # Auto-imported Vue components
│   ├── composables/            # Auto-imported composable functions
│   ├── layouts/                # Layout components
│   ├── middleware/             # Route middleware
│   ├── pages/                  # File-based routing pages
│   └── plugins/                # Vue plugins
├── server/
│   ├── api/                    # Auto-routed API endpoints
│   ├── routes/                 # Custom server routes
│   └── utils/                  # Server utilities
├── prisma/
│   ├── schema.prisma           # Database schema
│   └── migrations/             # Database migrations
├── test/
│   ├── unit/                   # Vitest unit tests
│   └── e2e/                    # Playwright E2E tests
├── nuxt.config.ts              # Nuxt configuration
└── package.json
```

## Key Changes from Nuxt 3

Nuxt 4 encourages using the `app/` directory to encapsulate all frontend code. When using this structure:

| Nuxt 3 | Nuxt 4 (app/) |
|--------|---------------|
| `pages/` | `app/pages/` |
| `components/` | `app/components/` |
| `composables/` | `app/composables/` |
| `layouts/` | `app/layouts/` |
| `assets/css/` | `app/assets/css/` |

## Auto-Import Behavior

**Important:** When using the `app/` directory structure, all composables in `app/composables/` are **automatically imported** by Nuxt. You do NOT need manual imports:

```typescript
// ✅ Correct - auto-imported in Nuxt 4
<script setup lang="ts">
const { data, loading, fetchOverview } = useAdminAnalytics()
</script>

// ❌ Not needed - avoid manual imports for local composables
<script setup lang="ts">
import { useAdminAnalytics } from '~/composables/useAdminAnalytics'
import { useAdminAnalytics } from '../../../composables/useAdminAnalytics'
import { useAdminAnalytics } from '#imports'
</script>
```

## Common Issue: Composable Not Found

### Problem

Error message: `useAdminAnalytics is not defined`

Or Vite error: `Failed to resolve import "~/composables/..." - Does the file exist?`

### Root Cause

When using `app/pages/` instead of root-level `pages/`, relative imports like `../../../composables/...` fail because:
1. The path resolution is different in the `app/` context
2. Manual imports conflict with Nuxt's auto-import system

### Solution

**Option 1: Use Auto-Imports (Recommended)**

Simply remove the manual import - Nuxt will auto-import composables from `app/composables/`:

```vue
<script setup lang="ts">
// No import needed - Nuxt auto-imports useAdminAnalytics
const { stats, loading, fetchOverview } = useAdminAnalytics()
</script>
```

**Option 2: Move Composables Directory**

If composables aren't being auto-imported, ensure they're in the correct location:

```
# ❌ Wrong location (won't auto-import with app/ structure)
project/
├── composables/
└── app/

# ✅ Correct location
project/
├── app/
│   └── composables/
```

**Option 3: Configure Nuxt**

Add explicit configuration in `nuxt.config.ts`:

```typescript
export default defineNuxtConfig({
  // If using app/ directory, Nuxt auto-detects it
  // But you can be explicit:
  srcDir: 'app/',

  // Or configure custom composable directories:
  imports: {
    dirs: ['composables', 'app/composables']
  }
})
```

## Best Practices

1. **Always use `app/composables/`** when using `app/pages/` - this ensures auto-imports work correctly

2. **Never manually import local composables** - let Nuxt handle it:
   ```typescript
   // ✅ Good
   const { data } = useFetch('/api/data')

   // ❌ Unnecessary
   import { useFetch } from '#app/composables/fetch'
   ```

3. **Keep composables at the root of `app/composables/`** - avoid deep nesting for auto-import to work reliably

4. **Use `ref`, `computed`, `watch` from Vue** - these are also auto-imported:
   ```typescript
   <script setup lang="ts">
   const count = ref(0)
   const doubled = computed(() => count.value * 2)
   </script>
   ```

5. **After moving the composables directory, restart the dev server** - Nuxt needs to regenerate the auto-import manifest:
   ```bash
   # Stop dev server, then:
   pnpm dev
   ```

## Debugging Auto-Imports

Check generated imports in `.nuxt/imports.d.ts`:

```bash
cat .nuxt/imports.d.ts | grep useAdminAnalytics
```

If your composable isn't listed:
1. Verify it's in `app/composables/`
2. Ensure it exports a named function
3. Restart the dev server
4. Delete `.nuxt/` and restart if needed

```bash
rm -rf .nuxt && pnpm dev
```
