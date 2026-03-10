<template>
  <NuxtLayout name="default">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h1 class="text-3xl font-bold mb-8">Contact Us</h1>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        <!-- Contact Info -->
        <div>
          <h2 class="text-xl font-semibold mb-4">Get in Touch</h2>
          <p class="text-gray-600 dark:text-gray-400 mb-6">
            Have questions or feedback? We'd love to hear from you. Fill out the form and we'll get back to you as soon as possible.
          </p>

          <div class="space-y-4">
            <div class="flex items-center gap-3">
              <UIcon name="i-lucide-mail" class="w-5 h-5 text-primary" />
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Email</p>
                <p class="font-medium">support@englishreading.com</p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <UIcon name="i-lucide-phone" class="w-5 h-5 text-primary" />
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Phone</p>
                <p class="font-medium">+86 400-123-4567</p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <UIcon name="i-lucide-clock" class="w-5 h-5 text-primary" />
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Business Hours</p>
                <p class="font-medium">Monday - Friday: 9:00 AM - 6:00 PM (CST)</p>
              </div>
            </div>
          </div>

          <!-- Social Links -->
          <div class="mt-8">
            <h3 class="text-lg font-semibold mb-4">Follow Us</h3>
            <div class="flex gap-3">
              <UButton icon="i-lucide-twitter" color="neutral" variant="ghost" size="lg" />
              <UButton icon="i-lucide-facebook" color="neutral" variant="ghost" size="lg" />
              <UButton icon="i-lucide-instagram" color="neutral" variant="ghost" size="lg" />
              <UButton icon="i-lucide-youtube" color="neutral" variant="ghost" size="lg" />
            </div>
          </div>
        </div>

        <!-- Contact Form -->
        <div>
          <UCard>
            <form @submit.prevent="handleSubmit" class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-1">Name</label>
                <UInput v-model="form.name" placeholder="Your name" required />
              </div>
              <div>
                <label class="block text-sm font-medium mb-1">Email</label>
                <UInput v-model="form.email" type="email" placeholder="your@email.com" required />
              </div>
              <div>
                <label class="block text-sm font-medium mb-1">Subject</label>
                <USelect
                  v-model="form.subject"
                  :items="subjectOptions"
                  placeholder="Select a subject"
                />
              </div>
              <div>
                <label class="block text-sm font-medium mb-1">Message</label>
                <UTextarea v-model="form.message" :rows="5" placeholder="Your message..." required />
              </div>
              <UButton type="submit" block :loading="submitting">
                Send Message
              </UButton>
            </form>
          </UCard>
        </div>
      </div>

      <!-- FAQ Quick Links -->
      <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-8">
        <h2 class="text-xl font-semibold mb-4">Frequently Asked Questions</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NuxtLink to="/membership#faq" class="flex items-center gap-2 text-primary hover:underline">
            <UIcon name="i-lucide-help-circle" class="w-5 h-5" />
            <span>How do I upgrade my plan?</span>
          </NuxtLink>
          <NuxtLink to="/membership#faq" class="flex items-center gap-2 text-primary hover:underline">
            <UIcon name="i-lucide-help-circle" class="w-5 h-5" />
            <span>What payment methods do you accept?</span>
          </NuxtLink>
          <NuxtLink to="/membership#faq" class="flex items-center gap-2 text-primary hover:underline">
            <UIcon name="i-lucide-help-circle" class="w-5 h-5" />
            <span>Can I cancel my subscription?</span>
          </NuxtLink>
          <NuxtLink to="/privacy" class="flex items-center gap-2 text-primary hover:underline">
            <UIcon name="i-lucide-help-circle" class="w-5 h-5" />
            <span>How is my data protected?</span>
          </NuxtLink>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const toast = useToast()

const form = ref({
  name: '',
  email: '',
  subject: '',
  message: ''
})

const submitting = ref(false)

const subjectOptions = [
  { label: 'General Inquiry', value: 'general' },
  { label: 'Technical Support', value: 'technical' },
  { label: 'Billing Question', value: 'billing' },
  { label: 'Feature Request', value: 'feature' },
  { label: 'Partnership', value: 'partnership' },
  { label: 'Other', value: 'other' }
]

const handleSubmit = async () => {
  submitting.value = true
  try {
    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1000))
    toast.add({
      title: 'Message Sent',
      description: 'Thank you for contacting us. We will get back to you soon.',
      color: 'success'
    })
    form.value = {
      name: '',
      email: '',
      subject: '',
      message: ''
    }
  } catch (error) {
    toast.add({
      title: 'Error',
      description: 'Failed to send message. Please try again.',
      color: 'error'
    })
  } finally {
    submitting.value = false
  }
}

useSeoMeta({
  title: 'Contact Us - English Reading',
  description: 'Contact English Reading App for support, feedback, or questions'
})
</script>