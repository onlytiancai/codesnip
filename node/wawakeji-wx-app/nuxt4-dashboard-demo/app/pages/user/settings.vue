<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 class="text-2xl font-bold mb-8">Settings</h1>

      <!-- Loading State -->
      <div v-if="loading && !preferences" class="flex justify-center py-12">
        <UIcon name="i-lucide-loader-2" class="w-8 h-8 animate-spin text-primary" />
      </div>

      <UTabs v-else :items="tabs" orientation="horizontal" class="gap-4">
        <template #profile>
          <UCard class="mt-6">
            <template #header>
              <h3 class="font-semibold">Profile Settings</h3>
            </template>
            <form class="space-y-4" @submit.prevent="saveProfile">
              <div class="flex items-center gap-4">
                <UAvatar :src="profileForm.avatar || undefined" :alt="profileForm.name || 'User'" size="xl" />
                <div>
                  <UButton size="sm" variant="outline" type="button">Change Avatar</UButton>
                  <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    JPG, PNG or GIF. Max 2MB.
                  </p>
                </div>
              </div>
              <UFormField label="Full Name" name="name">
                <UInput v-model="profileForm.name" />
              </UFormField>
              <UFormField label="Email" name="email">
                <UInput :model-value="profileForm.email" type="email" disabled />
              </UFormField>
              <UFormField label="Bio" name="bio">
                <UTextarea v-model="profileForm.bio" placeholder="Tell us about yourself..." :rows="3" />
              </UFormField>
              <div class="pt-4">
                <UButton type="submit" :loading="saving">Save Changes</UButton>
              </div>
            </form>
          </UCard>
        </template>

        <template #preferences>
          <div class="space-y-6 mt-6">
            <UCard>
              <template #header>
                <h3 class="font-semibold">Reading Preferences</h3>
              </template>
              <form class="space-y-4" @submit.prevent="savePreferences">
                <UFormField label="English Level" name="level">
                  <USelect
                    v-model="preferencesForm.englishLevel"
                    :items="[
                      { label: 'Beginner', value: 'beginner' },
                      { label: 'Intermediate', value: 'intermediate' },
                      { label: 'Advanced', value: 'advanced' }
                    ]"
                    class="w-64"
                  />
                </UFormField>
                <UFormField label="Daily Reading Goal" name="goal">
                  <USelect
                    v-model="preferencesForm.dailyGoal"
                    :items="[
                      { label: '5 minutes', value: 5 },
                      { label: '10 minutes', value: 10 },
                      { label: '15 minutes', value: 15 },
                      { label: '30 minutes', value: 30 }
                    ]"
                    class="w-64"
                  />
                </UFormField>
                <UFormField label="Default Audio Speed" name="speed">
                  <USelect
                    v-model="preferencesForm.audioSpeed"
                    :items="[
                      { label: '0.5x', value: 0.5 },
                      { label: '0.75x', value: 0.75 },
                      { label: '1.0x (Normal)', value: 1.0 },
                      { label: '1.25x', value: 1.25 },
                      { label: '1.5x', value: 1.5 }
                    ]"
                    class="w-64"
                  />
                </UFormField>
                <div class="pt-4">
                  <UButton type="submit" :loading="saving">Save Preferences</UButton>
                </div>
              </form>
            </UCard>

            <UCard>
              <template #header>
                <h3 class="font-semibold">Interest Categories</h3>
              </template>
              <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
                Select categories you're interested in for personalized recommendations
              </p>
              <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div
                  v-for="interest in availableInterests"
                  :key="interest.id"
                  :class="[
                    'p-3 rounded-lg border-2 cursor-pointer transition text-center',
                    preferencesForm.interests.includes(interest.id)
                      ? 'border-primary bg-primary/5'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary/50'
                  ]"
                  @click="toggleInterest(interest.id)"
                >
                  <UIcon :name="interest.icon" class="w-5 h-5 mx-auto mb-1" />
                  <span class="text-sm">{{ interest.name }}</span>
                </div>
              </div>
            </UCard>
          </div>
        </template>

        <template #appearance>
          <UCard class="mt-6">
            <template #header>
              <h3 class="font-semibold">Appearance</h3>
            </template>
            <form class="space-y-6" @submit.prevent="savePreferences">
              <div>
                <h4 class="font-medium mb-3">Theme</h4>
                <div class="flex gap-3">
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      preferencesForm.theme === 'light' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="preferencesForm.theme = 'light'"
                  >
                    <UIcon name="i-lucide-sun" class="w-6 h-6 mx-auto mb-2" />
                    <span class="text-sm font-medium">Light</span>
                  </div>
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      preferencesForm.theme === 'dark' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="preferencesForm.theme = 'dark'"
                  >
                    <UIcon name="i-lucide-moon" class="w-6 h-6 mx-auto mb-2" />
                    <span class="text-sm font-medium">Dark</span>
                  </div>
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      preferencesForm.theme === 'system' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="preferencesForm.theme = 'system'"
                  >
                    <UIcon name="i-lucide-monitor" class="w-6 h-6 mx-auto mb-2" />
                    <span class="text-sm font-medium">System</span>
                  </div>
                </div>
              </div>
              <div>
                <h4 class="font-medium mb-3">Font Size</h4>
                <div class="flex items-center gap-4">
                  <span class="text-sm">A</span>
                  <USlider v-model="preferencesForm.fontSize" :min="12" :max="24" class="flex-1" />
                  <span class="text-lg">A</span>
                </div>
                <p class="text-sm text-gray-500 mt-2">Current: {{ preferencesForm.fontSize }}px</p>
              </div>
              <div class="pt-4">
                <UButton type="submit" :loading="saving">Save Appearance</UButton>
              </div>
            </form>
          </UCard>
        </template>

        <template #notifications>
          <UCard class="mt-6">
            <template #header>
              <h3 class="font-semibold">Notification Settings</h3>
            </template>
            <form class="space-y-4" @submit.prevent="savePreferences">
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Reading Reminders</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Get daily reminders to maintain your streak
                  </p>
                </div>
                <USwitch v-model="preferencesForm.reminderEnabled" />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">New Articles</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Be notified when new articles match your interests
                  </p>
                </div>
                <USwitch v-model="preferencesForm.newArticleNotify" />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Vocabulary Review</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Spaced repetition reminders for word review
                  </p>
                </div>
                <USwitch v-model="preferencesForm.vocabReviewNotify" />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Marketing Emails</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Receive news and special offers
                  </p>
                </div>
                <USwitch v-model="preferencesForm.marketingEmails" />
              </div>
              <div class="pt-4">
                <UButton type="submit" :loading="saving">Save Notifications</UButton>
              </div>
            </form>
          </UCard>
        </template>

        <template #account>
          <div class="space-y-6 mt-6">
            <UCard>
              <template #header>
                <h3 class="font-semibold">Change Password</h3>
              </template>
              <form class="space-y-4" @submit.prevent="handleChangePassword">
                <UFormField label="Current Password" name="currentPassword">
                  <UInput v-model="passwordForm.currentPassword" type="password" class="w-64" />
                </UFormField>
                <UFormField label="New Password" name="newPassword">
                  <UInput v-model="passwordForm.newPassword" type="password" class="w-64" />
                </UFormField>
                <UFormField label="Confirm New Password" name="confirmPassword">
                  <UInput v-model="passwordForm.confirmPassword" type="password" class="w-64" />
                </UFormField>
                <UButton type="submit" :loading="changingPassword">Update Password</UButton>
              </form>
            </UCard>

            <UCard>
              <template #header>
                <h3 class="font-semibold text-red-500">Danger Zone</h3>
              </template>
              <div class="space-y-4">
                <div class="flex items-center justify-between">
                  <div>
                    <p class="font-medium">Delete Account</p>
                    <p class="text-sm text-gray-500 dark:text-gray-400">
                      Permanently delete your account and all data
                    </p>
                  </div>
                  <UButton color="error" variant="soft" @click="showDeleteModal = true">
                    Delete Account
                  </UButton>
                </div>
              </div>
            </UCard>
          </div>
        </template>
      </UTabs>

      <!-- Delete Account Modal -->
      <UModal v-model:open="showDeleteModal">
        <template #content>
          <UCard>
            <template #header>
              <h3 class="text-lg font-semibold text-red-500">Delete Account</h3>
            </template>
            <p class="text-gray-600 dark:text-gray-300 mb-4">
              This action is irreversible. All your data will be permanently deleted, including:
            </p>
            <ul class="list-disc list-inside text-sm text-gray-500 dark:text-gray-400 mb-4">
              <li>Your reading history</li>
              <li>Your bookmarks</li>
              <li>Your vocabulary list</li>
              <li>Your preferences</li>
            </ul>
            <UFormField label="Type 'DELETE MY ACCOUNT' to confirm" name="confirm">
              <UInput v-model="deleteConfirm" placeholder="DELETE MY ACCOUNT" />
            </UFormField>
            <template #footer>
              <div class="flex justify-end gap-2">
                <UButton variant="outline" @click="showDeleteModal = false">Cancel</UButton>
                <UButton
                  color="error"
                  :disabled="deleteConfirm !== 'DELETE MY ACCOUNT'"
                  :loading="deleting"
                  @click="handleDeleteAccount"
                >
                  Delete My Account
                </UButton>
              </div>
            </template>
          </UCard>
        </template>
      </UModal>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
definePageMeta({
  middleware: 'auth'
})

const { preferences, loading, fetchPreferences, updatePreferences, changePassword, deleteAccount } = useUserPreferences()
const { profile, fetchProfile, updateProfile } = useUserProfile()
const { clear } = useUserSession()
const router = useRouter()
const toast = useToast()

const tabs = [
  { label: 'Profile', slot: 'profile' },
  { label: 'Preferences', slot: 'preferences' },
  { label: 'Appearance', slot: 'appearance' },
  { label: 'Notifications', slot: 'notifications' },
  { label: 'Account', slot: 'account' }
]

const saving = ref(false)
const changingPassword = ref(false)
const deleting = ref(false)
const showDeleteModal = ref(false)
const deleteConfirm = ref('')

const profileForm = reactive({
  name: '',
  email: '',
  avatar: '',
  bio: ''
})

const preferencesForm = reactive({
  englishLevel: 'intermediate',
  dailyGoal: 10,
  audioSpeed: 1.0,
  theme: 'system',
  fontSize: 16,
  interests: [] as string[],
  reminderEnabled: true,
  newArticleNotify: true,
  vocabReviewNotify: false,
  marketingEmails: false
})

const passwordForm = reactive({
  currentPassword: '',
  newPassword: '',
  confirmPassword: ''
})

const availableInterests = [
  { id: 'technology', name: 'Technology', icon: 'i-lucide-cpu' },
  { id: 'science', name: 'Science', icon: 'i-lucide-flask-conical' },
  { id: 'business', name: 'Business', icon: 'i-lucide-briefcase' },
  { id: 'health', name: 'Health', icon: 'i-lucide-heart-pulse' },
  { id: 'culture', name: 'Culture', icon: 'i-lucide-globe' },
  { id: 'travel', name: 'Travel', icon: 'i-lucide-plane' }
]

onMounted(async () => {
  await Promise.all([
    fetchPreferences(),
    fetchProfile()
  ])

  // Initialize forms with fetched data
  if (preferences.value) {
    Object.assign(preferencesForm, {
      englishLevel: preferences.value.englishLevel,
      dailyGoal: preferences.value.dailyGoal,
      audioSpeed: preferences.value.audioSpeed,
      theme: preferences.value.theme,
      fontSize: preferences.value.fontSize,
      interests: preferences.value.interests || [],
      reminderEnabled: preferences.value.reminderEnabled,
      newArticleNotify: preferences.value.newArticleNotify,
      vocabReviewNotify: preferences.value.vocabReviewNotify,
      marketingEmails: preferences.value.marketingEmails
    })
  }

  if (profile.value) {
    Object.assign(profileForm, {
      name: profile.value.user.name || '',
      email: profile.value.user.email,
      avatar: profile.value.user.avatar || '',
      bio: profile.value.user.bio || ''
    })
  }
})

const toggleInterest = (id: string) => {
  const index = preferencesForm.interests.indexOf(id)
  if (index === -1) {
    preferencesForm.interests.push(id)
  } else {
    preferencesForm.interests.splice(index, 1)
  }
}

const saveProfile = async () => {
  saving.value = true
  try {
    await updateProfile({
      name: profileForm.name,
      bio: profileForm.bio || null
    })
    toast.add({
      title: 'Profile updated',
      color: 'success'
    })
  } catch (error) {
    toast.add({
      title: 'Failed to update profile',
      color: 'error'
    })
  } finally {
    saving.value = false
  }
}

const savePreferences = async () => {
  saving.value = true
  try {
    await updatePreferences({
      englishLevel: preferencesForm.englishLevel as 'beginner' | 'intermediate' | 'advanced',
      dailyGoal: preferencesForm.dailyGoal,
      audioSpeed: preferencesForm.audioSpeed,
      theme: preferencesForm.theme as 'light' | 'dark' | 'system',
      fontSize: preferencesForm.fontSize,
      interests: preferencesForm.interests,
      reminderEnabled: preferencesForm.reminderEnabled,
      newArticleNotify: preferencesForm.newArticleNotify,
      vocabReviewNotify: preferencesForm.vocabReviewNotify,
      marketingEmails: preferencesForm.marketingEmails
    })
    toast.add({
      title: 'Preferences saved',
      color: 'success'
    })
  } catch (error) {
    toast.add({
      title: 'Failed to save preferences',
      color: 'error'
    })
  } finally {
    saving.value = false
  }
}

const handleChangePassword = async () => {
  if (passwordForm.newPassword !== passwordForm.confirmPassword) {
    toast.add({
      title: 'Passwords do not match',
      color: 'error'
    })
    return
  }

  if (passwordForm.newPassword.length < 6) {
    toast.add({
      title: 'Password too short',
      description: 'Password must be at least 6 characters',
      color: 'error'
    })
    return
  }

  changingPassword.value = true
  try {
    await changePassword(passwordForm.currentPassword, passwordForm.newPassword)
    toast.add({
      title: 'Password updated',
      color: 'success'
    })
    passwordForm.currentPassword = ''
    passwordForm.newPassword = ''
    passwordForm.confirmPassword = ''
  } catch (error: any) {
    toast.add({
      title: 'Failed to change password',
      description: error.data?.message || 'Please check your current password',
      color: 'error'
    })
  } finally {
    changingPassword.value = false
  }
}

const handleDeleteAccount = async () => {
  deleting.value = true
  try {
    await deleteAccount()
    await clear()
    toast.add({
      title: 'Account deleted',
      description: 'Your account has been permanently deleted',
      color: 'success'
    })
    router.push('/')
  } catch (error) {
    toast.add({
      title: 'Failed to delete account',
      color: 'error'
    })
  } finally {
    deleting.value = false
    showDeleteModal.value = false
  }
}
</script>