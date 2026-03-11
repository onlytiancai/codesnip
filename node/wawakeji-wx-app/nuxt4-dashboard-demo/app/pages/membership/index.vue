<template>
  <NuxtLayout name="default">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <!-- Header -->
      <div class="text-center mb-12">
        <UBadge color="warning" variant="subtle" class="mb-4">
          <UIcon name="i-lucide-crown" class="w-4 h-4 mr-1" />
          Premium Membership
        </UBadge>
        <h1 class="text-4xl font-bold mb-4">Unlock Your Full Potential</h1>
        <p class="text-lg text-gray-500 dark:text-gray-400 max-w-2xl mx-auto">
          Get unlimited access to all articles, vocabulary tools, and premium features to accelerate your English learning.
        </p>
      </div>

      <!-- Current Plan Status -->
      <ClientOnly>
        <div v-if="loggedIn && membership" class="max-w-md mx-auto mb-8">
          <UCard class="bg-primary/5 border-primary/20">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Current Plan</p>
                <p class="text-lg font-semibold capitalize">{{ membership.plan }}</p>
              </div>
              <UBadge :color="membership.plan === 'free' ? 'neutral' : 'primary'">
                {{ membership.plan === 'free' ? 'Free' : 'Premium' }}
              </UBadge>
            </div>
            <p v-if="membership.endDate" class="text-sm text-gray-500 mt-2">
              Valid until {{ formatDate(membership.endDate) }}
            </p>
          </UCard>
        </div>
      </ClientOnly>

      <!-- Pricing Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <!-- Free Plan -->
        <UCard>
          <template #header>
            <h3 class="text-lg font-semibold">Free</h3>
            <div class="mt-4">
              <span class="text-4xl font-bold">$0</span>
              <span class="text-gray-500 dark:text-gray-400">/month</span>
            </div>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>5 articles per month</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Basic audio playback</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Reading progress tracking</span>
            </li>
            <li class="flex items-center gap-2 text-gray-400">
              <UIcon name="i-lucide-x" class="w-5 h-5" />
              <span>Limited vocabulary (10 words)</span>
            </li>
            <li class="flex items-center gap-2 text-gray-400">
              <UIcon name="i-lucide-x" class="w-5 h-5" />
              <span>Ads included</span>
            </li>
          </ul>
          <template #footer>
            <ClientOnly>
              <UButton
                block
                variant="outline"
                :loading="upgrading === 'free'"
                :disabled="membership?.plan === 'free'"
                @click="handleDowngrade"
              >
                {{ membership?.plan === 'free' ? 'Current Plan' : 'Downgrade' }}
              </UButton>
            </ClientOnly>
          </template>
        </UCard>

        <!-- Pro Plan -->
        <UCard class="ring-2 ring-primary relative overflow-visible">
          <div class="absolute -top-3 left-1/2 -translate-x-1/2 z-10">
            <UBadge color="primary">Most Popular</UBadge>
          </div>
          <template #header>
            <h3 class="text-lg font-semibold">Pro</h3>
            <div class="mt-4">
              <span class="text-4xl font-bold">$9</span>
              <span class="text-gray-500 dark:text-gray-400">/month</span>
            </div>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Billed monthly
            </p>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Unlimited articles</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>HD audio with variable speed</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Full vocabulary features</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Spaced repetition system</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>No ads</span>
            </li>
          </ul>
          <template #footer>
            <ClientOnly>
              <UButton
                block
                color="primary"
                :loading="upgrading === 'pro'"
                :disabled="membership?.plan === 'premium'"
                @click="handleUpgrade('pro')"
              >
                {{ membership?.plan === 'premium' ? 'Current Plan' : membership?.plan === 'annual' ? 'Downgrade to Pro' : 'Upgrade to Pro' }}
              </UButton>
            </ClientOnly>
          </template>
        </UCard>

        <!-- Annual Plan -->
        <UCard>
          <template #header>
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold">Annual</h3>
              <UBadge color="success" variant="subtle">Save 40%</UBadge>
            </div>
            <div class="mt-4">
              <span class="text-4xl font-bold">$65</span>
              <span class="text-gray-500 dark:text-gray-400">/year</span>
            </div>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
              $5.42/month, billed annually
            </p>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Everything in Pro</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Priority support</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Early access to new features</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Exclusive content</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>2 months free</span>
            </li>
          </ul>
          <template #footer>
            <UButton
              block
              variant="outline"
              :loading="upgrading === 'annual'"
              :disabled="membership?.plan === 'annual'"
              @click="handleUpgrade('annual')"
            >
              {{ membership?.plan === 'annual' ? 'Current Plan' : 'Get Annual Plan' }}
            </UButton>
          </template>
        </UCard>
      </div>

      <!-- Feature Comparison -->
      <div class="mb-12">
        <h2 class="text-2xl font-bold text-center mb-6">Feature Comparison</h2>
        <UCard>
          <UTable :data="featureComparison" :columns="columns" class="w-full">
            <template #feature-cell="{ row }">
              <span class="font-medium">{{ row.original.feature }}</span>
            </template>
            <template #free-cell="{ row }">
              <UIcon v-if="row.original.free === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.free === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.free }}</span>
            </template>
            <template #pro-cell="{ row }">
              <UIcon v-if="row.original.pro === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.pro === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.pro }}</span>
            </template>
            <template #annual-cell="{ row }">
              <UIcon v-if="row.original.annual === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.annual === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.annual }}</span>
            </template>
          </UTable>
        </UCard>
      </div>

      <!-- FAQ -->
      <div class="max-w-3xl mx-auto">
        <h2 class="text-2xl font-bold text-center mb-8">Frequently Asked Questions</h2>
        <UAccordion :items="faqItems" class="space-y-4" />
      </div>

      <!-- Trust Badges -->
      <div class="flex flex-wrap justify-center gap-8 mt-12 pt-12 border-t border-gray-200 dark:border-gray-700">
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-shield-check" class="w-5 h-5" />
          <span>Secure Payment</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-refresh-ccw" class="w-5 h-5" />
          <span>30-day Money Back</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-headphones" class="w-5 h-5" />
          <span>24/7 Support</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-credit-card" class="w-5 h-5" />
          <span>Cancel Anytime</span>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const { loggedIn, user } = useUserSession()
const toast = useToast()
const router = useRouter()

const upgrading = ref<string | null>(null)
const membership = ref<any>(null)

// Fetch user's membership
watchEffect(async () => {
  if (loggedIn.value && user.value) {
    try {
      const profile = await $fetch('/api/user/profile')
      membership.value = profile.membership
    } catch (e) {
      // Ignore
    }
  }
})

const handleDowngrade = async () => {
  if (!loggedIn.value) {
    toast.add({
      title: 'Please login',
      description: 'You need to login to change your membership',
      color: 'warning'
    })
    router.push('/login')
    return
  }

  upgrading.value = 'free'
  try {
    await $fetch('/api/user/membership', {
      method: 'POST',
      body: { plan: 'free' }
    })
    toast.add({
      title: 'Success',
      description: 'You have successfully downgraded to free plan',
      color: 'success'
    })
    // Refresh membership
    const profile = await $fetch('/api/user/profile')
    membership.value = profile.membership
  } catch (error) {
    toast.add({
      title: 'Error',
      description: 'Failed to downgrade membership. Please try again.',
      color: 'error'
    })
  } finally {
    upgrading.value = null
  }
}

const handleUpgrade = async (plan: string) => {
  if (!loggedIn.value) {
    toast.add({
      title: 'Please login',
      description: 'You need to login to upgrade your membership',
      color: 'warning'
    })
    router.push('/login')
    return
  }

  upgrading.value = plan
  try {
    await $fetch('/api/user/membership', {
      method: 'POST',
      body: { plan }
    })
    toast.add({
      title: 'Success',
      description: `You have successfully upgraded to ${plan} plan!`,
      color: 'success'
    })
    // Refresh membership
    const profile = await $fetch('/api/user/profile')
    membership.value = profile.membership
  } catch (error) {
    toast.add({
      title: 'Error',
      description: 'Failed to upgrade membership. Please try again.',
      color: 'error'
    })
  } finally {
    upgrading.value = null
  }
}

const formatDate = (date: string | Date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

const columns = [
  { id: 'feature', header: 'Feature', size: 200 },
  { id: 'free', header: 'Free' },
  { id: 'pro', header: 'Pro' },
  { id: 'annual', header: 'Annual' }
]

const featureComparison = [
  { feature: 'Articles per month', free: '5', pro: 'Unlimited', annual: 'Unlimited' },
  { feature: 'Audio quality', free: 'Standard', pro: 'HD', annual: 'HD' },
  { feature: 'Audio playback speed', free: false, pro: true, annual: true },
  { feature: 'Vocabulary limit', free: '10 words', pro: 'Unlimited', annual: 'Unlimited' },
  { feature: 'Spaced repetition', free: false, pro: true, annual: true },
  { feature: 'Offline access', free: false, pro: true, annual: true },
  { feature: 'Reading progress sync', free: true, pro: true, annual: true },
  { feature: 'Bookmark articles', free: true, pro: true, annual: true },
  { feature: 'Highlight & notes', free: false, pro: true, annual: true },
  { feature: 'Vocabulary export', free: false, pro: true, annual: true },
  { feature: 'Dark mode', free: true, pro: true, annual: true },
  { feature: 'Ad-free experience', free: false, pro: true, annual: true },
  { feature: 'Priority support', free: false, pro: false, annual: true },
  { feature: 'Early access to features', free: false, pro: false, annual: true },
  { feature: 'Exclusive content', free: false, pro: false, annual: true }
]

const faqItems = [
  {
    label: 'Can I cancel my subscription anytime?',
    content: 'Yes, you can cancel your subscription at any time. Your access will continue until the end of your billing period.'
  },
  {
    label: 'What payment methods do you accept?',
    content: 'We accept WeChat Pay, Alipay, and major credit cards including Visa, Mastercard, and American Express.'
  },
  {
    label: 'Is there a refund policy?',
    content: 'Yes, we offer a 30-day money-back guarantee. If you are not satisfied, contact us for a full refund.'
  },
  {
    label: 'Can I switch between plans?',
    content: 'Yes, you can upgrade or downgrade your plan at any time. The changes will take effect at your next billing cycle.'
  }
]
</script>