<template>
  <NuxtLayout name="admin">
    <div>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-bold">Dictionary</h2>
        <UButton icon="i-lucide-plus" @click="openCreateModal">
          Add Word
        </UButton>
      </div>

      <!-- Stats Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <UCard>
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">Total Words</p>
              <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.total || 0) }}</p>
            </div>
            <div class="bg-blue-500 p-3 rounded-lg">
              <UIcon name="i-lucide-book" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>

        <UCard>
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">With Phonetic</p>
              <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.withPhonetic || 0) }}</p>
            </div>
            <div class="bg-green-500 p-3 rounded-lg">
              <UIcon name="i-lucide-volume-2" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>

        <UCard>
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">Collins 5-Star</p>
              <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.collins5Star || 0) }}</p>
            </div>
            <div class="bg-yellow-500 p-3 rounded-lg">
              <UIcon name="i-lucide-star" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>

        <UCard>
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm text-gray-500 dark:text-gray-400">Oxford 3000</p>
              <p class="text-2xl font-bold mt-1">{{ formatNumber(stats?.oxford3000 || 0) }}</p>
            </div>
            <div class="bg-purple-500 p-3 rounded-lg">
              <UIcon name="i-lucide-graduation-cap" class="w-6 h-6 text-white" />
            </div>
          </div>
        </UCard>
      </div>

      <!-- Tag Stats -->
      <UCard class="mb-6">
        <template #header>
          <h3 class="text-lg font-semibold">Words by Level/Exam</h3>
        </template>
        <div class="flex flex-wrap gap-2">
          <UBadge
            v-for="tag in stats?.tagStats"
            :key="tag.tag"
            :color="filters.tag === tag.tag ? 'primary' : 'neutral'"
            variant="subtle"
            class="cursor-pointer"
            @click="filterByTag(tag.tag)"
          >
            {{ tag.label }}: {{ formatNumber(tag.count) }}
          </UBadge>
          <UBadge
            v-if="filters.tag"
            color="error"
            variant="subtle"
            class="cursor-pointer"
            @click="filterByTag('')"
          >
            Clear Filter
          </UBadge>
        </div>
      </UCard>

      <!-- Filters -->
      <UCard class="mb-6">
        <div class="flex flex-wrap gap-4 items-end">
          <div class="flex-1 min-w-[200px]">
            <UFormField label="Search">
              <UInput
                v-model="filters.search"
                placeholder="Search words..."
                icon="i-lucide-search"
                @keyup.enter="applyFilters"
              />
            </UFormField>
          </div>
          <div class="w-48">
            <UFormField label="Phonetic">
              <USelect
                v-model="filters.hasPhonetic"
                :items="[
                  { label: 'All', value: 'all' },
                  { label: 'With Phonetic', value: 'true' },
                  { label: 'No Phonetic', value: 'false' }
                ]"
              />
            </UFormField>
          </div>
          <div class="w-48">
            <UFormField label="Translation">
              <USelect
                v-model="filters.hasTranslation"
                :items="[
                  { label: 'All', value: 'all' },
                  { label: 'With Translation', value: 'true' },
                  { label: 'No Translation', value: 'false' }
                ]"
              />
            </UFormField>
          </div>
          <div class="w-40">
            <UFormField label="Sort By">
              <USelect
                v-model="filters.sortBy"
                :items="[
                  { label: 'Word (A-Z)', value: 'word' },
                  { label: 'Collins ★', value: 'collins' },
                  { label: 'BNC Freq', value: 'bnc' },
                  { label: 'Corpus Freq', value: 'frq' }
                ]"
                @update:model-value="handleSortChange"
              />
            </UFormField>
          </div>
          <UButton @click="applyFilters">Apply</UButton>
        </div>
      </UCard>

      <!-- Loading State -->
      <div v-if="loading" class="flex items-center justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <!-- Words Table -->
      <UCard v-else>
        <UTable :data="words" :columns="columns">
          <template #word-cell="{ row }">
            <div>
              <p class="font-medium">{{ row.original.word }}</p>
              <p v-if="row.original.phonetic" class="text-xs text-gray-500">{{ row.original.phonetic }}</p>
            </div>
          </template>
          <template #translation-cell="{ row }">
            <p class="text-sm line-clamp-2 max-w-xs">{{ row.original.translation || '-' }}</p>
          </template>
          <template #pos-cell="{ row }">
            <span class="text-sm">{{ row.original.pos || '-' }}</span>
          </template>
          <template #collins-cell="{ row }">
            <div v-if="row.original.collins" class="flex gap-0.5">
              <UIcon
                v-for="i in 5"
                :key="i"
                :name="i <= row.original.collins ? 'i-lucide-star' : 'i-lucide-star'"
                :class="i <= row.original.collins ? 'text-yellow-400' : 'text-gray-300'"
                class="w-3 h-3"
              />
            </div>
            <span v-else>-</span>
          </template>
          <template #tag-cell="{ row }">
            <div v-if="row.original.tag" class="flex flex-wrap gap-1">
              <UBadge
                v-for="t in row.original.tag.split(/\s+/).slice(0, 3)"
                :key="t"
                variant="subtle"
                size="xs"
              >
                {{ getTagLabel(t) }}
              </UBadge>
              <UBadge v-if="row.original.tag.split(/\s+/).length > 3" variant="subtle" size="xs">
                +{{ row.original.tag.split(/\s+/).length - 3 }}
              </UBadge>
            </div>
            <span v-else>-</span>
          </template>
          <template #actions-cell="{ row }">
            <div class="flex items-center gap-1">
              <UButton
                icon="i-lucide-edit"
                color="neutral"
                variant="ghost"
                size="xs"
                @click="openEditModal(row.original)"
              />
              <UButton
                icon="i-lucide-trash-2"
                color="error"
                variant="ghost"
                size="xs"
                @click="confirmDelete(row.original)"
              />
            </div>
          </template>
        </UTable>

        <!-- Pagination -->
        <div class="flex items-center justify-between mt-4 pt-4 border-t">
          <p class="text-sm text-gray-500">
            Showing {{ (pagination.page - 1) * pagination.limit + 1 }}-{{ Math.min(pagination.page * pagination.limit, pagination.total) }} of {{ formatNumber(pagination.total) }} words
          </p>
          <UPagination
            v-model:page="currentPage"
            :total="pagination.total"
            :items-per-page="pagination.limit"
            :sibling-count="2"
          />
        </div>
      </UCard>

      <!-- Add/Edit Modal -->
      <UModal
        v-model:open="showModal"
        :title="editingWord ? 'Edit Word' : 'Add Word'"
        description="Configure the word details below"
      >
        <template #body>
          <div class="space-y-4">
            <UFormField v-if="!editingWord" label="Word" name="word" required>
              <UInput v-model="wordForm.word" placeholder="Enter word" />
            </UFormField>
            <UFormField label="Phonetic" name="phonetic">
              <UInput v-model="wordForm.phonetic" placeholder="/fəˈnetɪk/" />
            </UFormField>
            <UFormField label="Definition (English)" name="definition">
              <UTextarea v-model="wordForm.definition" placeholder="English definition" :rows="2" />
            </UFormField>
            <UFormField label="Translation (Chinese)" name="translation">
              <UTextarea v-model="wordForm.translation" placeholder="中文翻译" :rows="2" />
            </UFormField>
            <UFormField label="Part of Speech" name="pos">
              <UInput v-model="wordForm.pos" placeholder="n./v./adj./adv." />
            </UFormField>
            <div class="grid grid-cols-2 gap-4">
              <UFormField label="Collins (1-5)" name="collins">
                <USelect
                  v-model="wordForm.collins"
                  :items="[
                    { label: '-', value: null },
                    { label: '1 Star', value: 1 },
                    { label: '2 Stars', value: 2 },
                    { label: '3 Stars', value: 3 },
                    { label: '4 Stars', value: 4 },
                    { label: '5 Stars', value: 5 }
                  ]"
                />
              </UFormField>
              <UFormField label="Oxford 3000" name="oxford">
                <USelect
                  v-model="wordForm.oxford"
                  :items="[
                    { label: 'No', value: 0 },
                    { label: 'Yes', value: 1 }
                  ]"
                />
              </UFormField>
            </div>
            <UFormField label="Tags" name="tag">
              <UInput v-model="wordForm.tag" placeholder="zk gk cet4 cet6 ielts toefl gre" />
              <template #hint>
                <span class="text-xs text-gray-500">Space-separated: zk/中考, gk/高考, cet4/四级, cet6/六级, ielts/雅思, toefl/托福, gre/GRE</span>
              </template>
            </UFormField>
            <UFormField label="Word Forms" name="exchange">
              <UInput v-model="wordForm.exchange" placeholder="p:went/d:gone/i:going/3:goes" />
              <template #hint>
                <span class="text-xs text-gray-500">Format: p:past, d:past_part, i:ing, 3:3rd_person, s:plural</span>
              </template>
            </UFormField>
          </div>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showModal = false">Cancel</UButton>
            <UButton color="primary" :loading="saving" @click="handleSave">
              {{ editingWord ? 'Update' : 'Create' }}
            </UButton>
          </div>
        </template>
      </UModal>

      <!-- Delete Confirmation -->
      <UModal v-model:open="showDeleteModal" title="Delete Word" description="Are you sure you want to delete this word?">
        <template #body>
          <p class="text-gray-500">
            This action cannot be undone. The word "{{ wordToDelete?.word }}" will be permanently deleted from the dictionary.
          </p>
        </template>
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton variant="outline" @click="showDeleteModal = false">Cancel</UButton>
            <UButton color="error" :loading="deleting" @click="handleDelete">
              Delete
            </UButton>
          </div>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'admin'
})

const {
  words,
  stats,
  loading,
  pagination,
  filters,
  fetchWords,
  fetchStats,
  createWord,
  updateWord,
  deleteWord,
  applyFilters
} = useAdminDictionary()

const showModal = ref(false)
const showDeleteModal = ref(false)
const editingWord = ref<any>(null)
const wordToDelete = ref<any>(null)
const saving = ref(false)
const deleting = ref(false)

const currentPage = ref(1)

const wordForm = ref({
  word: '',
  phonetic: '',
  definition: '',
  translation: '',
  pos: '',
  collins: null as number | null,
  oxford: 0,
  tag: '',
  exchange: ''
})

// Handle sort change - frequency fields should default to descending
const handleSortChange = (value: string) => {
  if (value === 'collins' || value === 'bnc' || value === 'frq') {
    filters.value.sortOrder = 'desc'
  } else {
    filters.value.sortOrder = 'asc'
  }
}

const columns = [
  { id: 'word', header: 'Word' },
  { id: 'translation', header: 'Translation' },
  { id: 'pos', header: 'POS' },
  { id: 'collins', header: 'Collins' },
  { id: 'tag', header: 'Tags' },
  { id: 'actions', header: '' }
]

const tagLabels: Record<string, string> = {
  'zk': '中考',
  'gk': '高考',
  'cet4': '四级',
  'cet6': '六级',
  'ielts': '雅思',
  'toefl': '托福',
  'gre': 'GRE',
  'ky': '考研',
  'bec': 'BEC',
  'tem4': '专四',
  'tem8': '专八'
}

const getTagLabel = (tag: string) => tagLabels[tag.toLowerCase()] || tag.toUpperCase()

const formatNumber = (num: number) => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K'
  return num.toString()
}

const openCreateModal = () => {
  editingWord.value = null
  wordForm.value = {
    word: '',
    phonetic: '',
    definition: '',
    translation: '',
    pos: '',
    collins: null,
    oxford: 0,
    tag: '',
    exchange: ''
  }
  showModal.value = true
}

const openEditModal = (word: any) => {
  editingWord.value = word
  wordForm.value = {
    word: word.word,
    phonetic: word.phonetic || '',
    definition: word.definition || '',
    translation: word.translation || '',
    pos: word.pos || '',
    collins: word.collins,
    oxford: word.oxford || 0,
    tag: word.tag || '',
    exchange: word.exchange || ''
  }
  showModal.value = true
}

const handleSave = async () => {
  saving.value = true
  try {
    if (editingWord.value) {
      await updateWord(editingWord.value.word, wordForm.value)
    } else {
      await createWord(wordForm.value)
    }
    showModal.value = false
  } finally {
    saving.value = false
  }
}

const confirmDelete = (word: any) => {
  wordToDelete.value = word
  showDeleteModal.value = true
}

const handleDelete = async () => {
  if (!wordToDelete.value) return
  deleting.value = true
  try {
    await deleteWord(wordToDelete.value.word)
    showDeleteModal.value = false
    wordToDelete.value = null
  } finally {
    deleting.value = false
  }
}

const filterByTag = (tag: string) => {
  filters.value.tag = tag
  currentPage.value = 1
  applyFilters()
}

watch(currentPage, (page) => {
  fetchWords(page)
})

onMounted(() => {
  fetchWords()
  fetchStats()
})
</script>