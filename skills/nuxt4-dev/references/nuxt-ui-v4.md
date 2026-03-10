# Nuxt UI v4 Development Notes

Nuxt UI v4 (currently v3.x evolving to v4) is a comprehensive UI component library built on top of Tailwind CSS v4 and Reka UI. This guide covers common pitfalls and best practices.

## Version Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| @nuxt/ui | ^4.5.0 | Component library (v4 evolving) |
| lucide icons | ^1.2.94 | Via @nuxt/icon, use `lucide-*` icon names |

**Important:** Nuxt UI v4 components require `title` and `description` props for modals (accessibility requirement from Reka UI).

## Installation

```bash
pnpm add @nuxt/ui
```

In `nuxt.config.ts`:

```typescript
export default defineNuxtConfig({
  modules: [
    ["@nuxt/ui", { fonts: false }]  // fonts: false to disable auto-fonts
  ]
})
```

## Common Issues and Solutions

### UTable: Column Definition Error

**Problem:** Error message: "Columns require an id when using a non-string header"

**Cause:** Nuxt UI v4 uses TanStack Table internally, which requires `id` property for column definitions, not `key`.

**Solution:** Use `id` instead of `key` in column definitions:

```js
// ❌ Wrong - causes error
const columns = [
  { key: 'name', header: 'Name' },
  { key: 'email', header: 'Email' }
]

// ✅ Correct
const columns = [
  { id: 'name', header: 'Name', accessorKey: 'name' },
  { id: 'email', header: 'Email', accessorKey: 'email' }
]
```

**Full example:**

```vue
<script setup lang="ts">
const columns = [
  { id: 'name', header: 'Name', accessorKey: 'name' },
  { id: 'email', header: 'Email', accessorKey: 'email' },
  { id: 'role', header: 'Role', accessorKey: 'role' },
  { id: 'actions', header: 'Actions' }
]

const data = [
  { name: 'John', email: 'john@example.com', role: 'Admin' },
  { name: 'Jane', email: 'jane@example.com', role: 'User' }
]
</script>

<template>
  <UTable :columns="columns" :data="data" />
</template>
```

### UModal: Accessibility Warnings

**Problem:** UModal component throws accessibility warnings:
- "`DialogContent` requires a `DialogTitle` for the component to be accessible"
- "Missing `Description` or `aria-describedby` for DialogContent"

**Cause:** Nuxt UI v4's UModal uses Reka UI internally, which requires both `title` and `description` props for screen reader accessibility.

**Solution:** Add `title` and `description` props to UModal:

```vue
<!-- ❌ Wrong - causes accessibility warnings -->
<UModal v-model:open="showModal">
  <UCard>
    <template #header>
      <h3>Edit Category</h3>
    </template>
    <div>...content...</div>
  </UCard>
</UModal>

<!-- ✅ Correct - with title and description props -->
<UModal
  v-model:open="showModal"
  :title="editingCategory ? 'Edit Category' : 'Add Category'"
  description="Configure the category details below"
>
  <template #body>
    <div>...content...</div>
  </template>
  <template #footer>
    <div>...buttons...</div>
  </template>
</UModal>
```

**Key points:**
- Use `title` prop instead of `#header` slot with `<h3>`
- Always provide a `description` prop for accessibility
- Use `#body` and `#footer` slots for modal content
- The `v-model:open` syntax is required for v4

## Commonly Used Components

### UButton

```vue
<UButton
  color="primary"
  variant="solid"
  size="md"
  :disabled="loading"
  :loading="loading"
  @click="handleClick"
>
  Click me
</UButton>
```

**Variants:** `solid`, `outline`, `soft`, `ghost`, `link`
**Colors:** `primary`, `neutral`, `error`, `success`, `warning`, `info`

### UCard

```vue
<UCard>
  <template #header>
    <h3 class="text-lg font-semibold">Card Title</h3>
  </template>

  <p>Card content goes here</p>

  <template #footer>
    <UButton>Action</UButton>
  </template>
</UCard>
```

### UInput

```vue
<UInput
  v-model="email"
  type="email"
  placeholder="Enter your email"
  :error="!!errors.email"
  @blur="validateEmail"
/>
```

### UForm

```vue
<script setup lang="ts">
const form = ref(null)
const formData = ref({
  email: '',
  password: ''
})

function submit() {
  form.value?.submit()
}
</script>

<template>
  <UForm ref="form" :state="formData" @submit="onSubmit">
    <UFormField label="Email" name="email">
      <UInput v-model="formData.email" type="email" />
    </UFormField>

    <UFormField label="Password" name="password">
      <UInput v-model="formData.password" type="password" />
    </UFormField>

    <UButton type="submit">Submit</UButton>
  </UForm>
</template>
```

### UTable

```vue
<script setup lang="ts">
const columns = [
  { id: 'name', header: 'Name', accessorKey: 'name' },
  { id: 'email', header: 'Email', accessorKey: 'email' },
  {
    id: 'actions',
    header: 'Actions',
    cell: ({ row }) => h('div', [
      h(UButton, {
        icon: 'lucide-edit',
        variant: 'ghost',
        onClick: () => editItem(row.original)
      }),
      h(UButton, {
        icon: 'lucide-trash',
        variant: 'ghost',
        onClick: () => deleteItem(row.original)
      })
    ])
  }
]

const data = ref([])
</script>

<template>
  <UTable :columns="columns" :data="data" />
</template>
```

## Icons

Nuxt UI v4 uses Nuxt Icon (Iconify). Use Lucide icons:

```vue
<UButton icon="lucide-user">Profile</UButton>
<UButton icon="lucide-plus">Add</UButton>
<UButton icon="lucide-trash" color="error" />
```

Common icons:
- `lucide-user` - User/profile
- `lucide-plus` - Add/create
- `lucide-trash` - Delete
- `lucide-edit` - Edit
- `lucide-check` - Confirm/success
- `lucide-x` - Close/cancel
- `lucide-search` - Search
- `lucide-filter` - Filter

## Dark Mode

Nuxt UI automatically supports dark mode. Use `dark:` variants:

```vue
<div class="bg-white dark:bg-gray-900">
  <p class="text-gray-900 dark:text-gray-100">
    Dark mode supported text
  </p>
</div>
```

## Customization

In `app/app.config.ts`:

```typescript
export default defineAppConfig({
  ui: {
    primary: 'blue',
    gray: 'cool',
    button: {
      default: {
        size: 'md'
      }
    }
  }
})
```
