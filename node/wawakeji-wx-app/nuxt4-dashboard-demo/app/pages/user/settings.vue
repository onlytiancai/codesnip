<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 class="text-2xl font-bold mb-8">Settings</h1>

      <UTabs :items="tabs" orientation="horizontal" class="gap-4">
        <template #profile>
          <UCard class="mt-6">
            <template #header>
              <h3 class="font-semibold">Profile Settings</h3>
            </template>
            <div class="space-y-4">
              <div class="flex items-center gap-4">
                <UAvatar src="https://avatars.githubusercontent.com/u/1?v=4" alt="User" size="xl" />
                <div>
                  <UButton size="sm" variant="outline">Change Avatar</UButton>
                  <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    JPG, PNG or GIF. Max 2MB.
                  </p>
                </div>
              </div>
              <UFormField label="Full Name" name="name">
                <UInput default-value="John Doe" />
              </UFormField>
              <UFormField label="Email" name="email">
                <UInput default-value="john.doe@example.com" type="email" />
              </UFormField>
              <UFormField label="Bio" name="bio">
                <UTextarea placeholder="Tell us about yourself..." :rows="3" />
              </UFormField>
              <div class="pt-4">
                <UButton>Save Changes</UButton>
              </div>
            </div>
          </UCard>
        </template>

        <template #preferences>
          <div class="space-y-6 mt-6">
            <UCard>
              <template #header>
                <h3 class="font-semibold">Reading Preferences</h3>
              </template>
              <div class="space-y-4">
                <UFormField label="English Level" name="level">
                  <USelect
                    :items="[
                      { label: 'Beginner', value: 'beginner' },
                      { label: 'Intermediate', value: 'intermediate' },
                      { label: 'Advanced', value: 'advanced' }
                    ]"
                    default-value="intermediate"
                    class="w-64"
                  />
                </UFormField>
                <UFormField label="Daily Reading Goal" name="goal">
                  <USelect
                    :items="[
                      { label: '5 minutes', value: '5' },
                      { label: '10 minutes', value: '10' },
                      { label: '15 minutes', value: '15' },
                      { label: '30 minutes', value: '30' }
                    ]"
                    default-value="10"
                    class="w-64"
                  />
                </UFormField>
                <UFormField label="Default Audio Speed" name="speed">
                  <USelect
                    :items="[
                      { label: '0.5x', value: '0.5' },
                      { label: '0.75x', value: '0.75' },
                      { label: '1.0x (Normal)', value: '1.0' },
                      { label: '1.25x', value: '1.25' },
                      { label: '1.5x', value: '1.5' }
                    ]"
                    default-value="1.0"
                    class="w-64"
                  />
                </UFormField>
              </div>
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
                  v-for="interest in interests"
                  :key="interest.id"
                  :class="[
                    'p-3 rounded-lg border-2 cursor-pointer transition text-center',
                    selectedInterests.includes(interest.id)
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
            <div class="space-y-6">
              <div>
                <h4 class="font-medium mb-3">Theme</h4>
                <div class="flex gap-3">
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      theme === 'light' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="theme = 'light'"
                  >
                    <UIcon name="i-lucide-sun" class="w-6 h-6 mx-auto mb-2" />
                    <span class="text-sm font-medium">Light</span>
                  </div>
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      theme === 'dark' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="theme = 'dark'"
                  >
                    <UIcon name="i-lucide-moon" class="w-6 h-6 mx-auto mb-2" />
                    <span class="text-sm font-medium">Dark</span>
                  </div>
                  <div
                    :class="[
                      'flex-1 p-4 rounded-lg border-2 cursor-pointer transition text-center',
                      theme === 'system' ? 'border-primary bg-primary/5' : 'border-gray-200 dark:border-gray-700'
                    ]"
                    @click="theme = 'system'"
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
                  <USlider :default-value="50" class="flex-1" />
                  <span class="text-lg">A</span>
                </div>
              </div>
            </div>
          </UCard>
        </template>

        <template #notifications>
          <UCard class="mt-6">
            <template #header>
              <h3 class="font-semibold">Notification Settings</h3>
            </template>
            <div class="space-y-4">
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Reading Reminders</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Get daily reminders to maintain your streak
                  </p>
                </div>
                <USwitch default-value />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">New Articles</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Be notified when new articles match your interests
                  </p>
                </div>
                <USwitch default-value />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Vocabulary Review</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Spaced repetition reminders for word review
                  </p>
                </div>
                <USwitch />
              </div>
              <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                <div>
                  <p class="font-medium">Marketing Emails</p>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Receive news and special offers
                  </p>
                </div>
                <USwitch />
              </div>
            </div>
          </UCard>
        </template>

        <template #account>
          <div class="space-y-6 mt-6">
            <UCard>
              <template #header>
                <h3 class="font-semibold">Change Password</h3>
              </template>
              <div class="space-y-4">
                <UFormField label="Current Password" name="currentPassword">
                  <UInput type="password" class="w-64" />
                </UFormField>
                <UFormField label="New Password" name="newPassword">
                  <UInput type="password" class="w-64" />
                </UFormField>
                <UFormField label="Confirm New Password" name="confirmPassword">
                  <UInput type="password" class="w-64" />
                </UFormField>
                <UButton>Update Password</UButton>
              </div>
            </UCard>

            <UCard>
              <template #header>
                <h3 class="font-semibold">Linked Accounts</h3>
              </template>
              <div class="space-y-3">
                <div class="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700">
                  <div class="flex items-center gap-3">
                    <UIcon name="i-lucide-message-circle" class="w-5 h-5 text-green-500" />
                    <span>WeChat</span>
                  </div>
                  <UButton size="sm" variant="outline">Link Account</UButton>
                </div>
              </div>
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
                  <UButton color="error" variant="soft">Delete Account</UButton>
                </div>
              </div>
            </UCard>
          </div>
        </template>
      </UTabs>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const tabs = [
  { label: 'Profile', slot: 'profile' },
  { label: 'Preferences', slot: 'preferences' },
  { label: 'Appearance', slot: 'appearance' },
  { label: 'Notifications', slot: 'notifications' },
  { label: 'Account', slot: 'account' }
]

const theme = ref('system')
const selectedInterests = ref(['technology', 'science', 'business'])

const interests = [
  { id: 'technology', name: 'Technology', icon: 'i-lucide-cpu' },
  { id: 'science', name: 'Science', icon: 'i-lucide-flask-conical' },
  { id: 'business', name: 'Business', icon: 'i-lucide-briefcase' },
  { id: 'health', name: 'Health', icon: 'i-lucide-heart-pulse' },
  { id: 'culture', name: 'Culture', icon: 'i-lucide-globe' },
  { id: 'travel', name: 'Travel', icon: 'i-lucide-plane' }
]

const toggleInterest = (id: string) => {
  const index = selectedInterests.value.indexOf(id)
  if (index === -1) {
    selectedInterests.value.push(id)
  } else {
    selectedInterests.value.splice(index, 1)
  }
}
</script>