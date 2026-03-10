import 'dotenv/config'
import { PrismaBetterSQLite3 } from '@prisma/adapter-better-sqlite3'
import { PrismaClient } from '../generated/prisma/client'
import bcrypt from 'bcryptjs'

const adapter = new PrismaBetterSQLite3({ url: process.env.DATABASE_URL! })
const prisma = new PrismaClient({ adapter })

async function main() {
  console.log('Seeding database...')

  // Create admin user
  const adminPassword = await bcrypt.hash('admin123', 10)
  const admin = await prisma.user.upsert({
    where: { email: 'admin@example.com' },
    update: {
      password: adminPassword, // Update password in case it changed
    },
    create: {
      email: 'admin@example.com',
      name: 'Admin User',
      password: adminPassword,
      role: 'ADMIN',
      avatar: 'https://avatars.githubusercontent.com/u/1?v=4',
      bio: 'System administrator'
    }
  })

  console.log('Created admin user:', admin.email)

  // Create test user
  const userPassword = await bcrypt.hash('user123', 10)
  const user = await prisma.user.upsert({
    where: { email: 'user@example.com' },
    update: {
      password: userPassword, // Update password in case it changed
    },
    create: {
      email: 'user@example.com',
      name: 'Test User',
      password: userPassword,
      role: 'USER',
      avatar: 'https://avatars.githubusercontent.com/u/2?v=4',
      bio: 'English learner'
    }
  })

  console.log('Created test user:', user.email)

  // Create user preferences for both users
  await prisma.userPreferences.upsert({
    where: { userId: admin.id },
    update: {},
    create: {
      userId: admin.id,
      englishLevel: 'advanced',
      dailyGoal: 15,
      audioSpeed: 1.0,
      theme: 'system',
      fontSize: 16,
      interests: JSON.stringify(['technology', 'science']),
      reminderEnabled: true,
      newArticleNotify: true,
      vocabReviewNotify: true,
      marketingEmails: false
    }
  })

  await prisma.userPreferences.upsert({
    where: { userId: user.id },
    update: {},
    create: {
      userId: user.id,
      englishLevel: 'intermediate',
      dailyGoal: 10,
      audioSpeed: 1.0,
      theme: 'system',
      fontSize: 16,
      interests: JSON.stringify(['technology', 'business', 'health']),
      reminderEnabled: true,
      newArticleNotify: true,
      vocabReviewNotify: false,
      marketingEmails: false
    }
  })

  console.log('Created user preferences')

  // Create memberships
  await prisma.membership.upsert({
    where: { userId: admin.id },
    update: {},
    create: {
      userId: admin.id,
      plan: 'premium',
      startDate: new Date('2026-01-01'),
      endDate: new Date('2026-12-31')
    }
  })

  await prisma.membership.upsert({
    where: { userId: user.id },
    update: {},
    create: {
      userId: user.id,
      plan: 'free',
      startDate: new Date('2026-01-15')
    }
  })

  console.log('Created memberships')

  // Create categories
  const categories = await Promise.all([
    prisma.category.upsert({
      where: { slug: 'technology' },
      update: {},
      create: {
        name: 'Technology',
        slug: 'technology',
        icon: 'i-lucide-cpu',
        color: '#3b82f6',
        description: 'Technology and programming articles'
      }
    }),
    prisma.category.upsert({
      where: { slug: 'science' },
      update: {},
      create: {
        name: 'Science',
        slug: 'science',
        icon: 'i-lucide-flask-conical',
        color: '#22c55e',
        description: 'Scientific discoveries and research'
      }
    }),
    prisma.category.upsert({
      where: { slug: 'business' },
      update: {},
      create: {
        name: 'Business',
        slug: 'business',
        icon: 'i-lucide-briefcase',
        color: '#a855f7',
        description: 'Business and entrepreneurship'
      }
    }),
    prisma.category.upsert({
      where: { slug: 'health' },
      update: {},
      create: {
        name: 'Health',
        slug: 'health',
        icon: 'i-lucide-heart-pulse',
        color: '#ef4444',
        description: 'Health and wellness'
      }
    })
  ])

  console.log('Created categories:', categories.length)

  // Create tags
  const tags = await Promise.all([
    prisma.tag.upsert({
      where: { slug: 'ai' },
      update: {},
      create: { name: 'AI', slug: 'ai', color: '#3b82f6' }
    }),
    prisma.tag.upsert({
      where: { slug: 'machine-learning' },
      update: {},
      create: { name: 'Machine Learning', slug: 'machine-learning', color: '#8b5cf6' }
    }),
    prisma.tag.upsert({
      where: { slug: 'startup' },
      update: {},
      create: { name: 'Startup', slug: 'startup', color: '#a855f7' }
    }),
    prisma.tag.upsert({
      where: { slug: 'healthcare' },
      update: {},
      create: { name: 'Healthcare', slug: 'healthcare', color: '#ef4444' }
    }),
    prisma.tag.upsert({
      where: { slug: 'climate' },
      update: {},
      create: { name: 'Climate', slug: 'climate', color: '#22c55e' }
    })
  ])

  console.log('Created tags:', tags.length)

  // Create sample article
  const article = await prisma.article.upsert({
    where: { slug: 'ai-in-healthcare' },
    update: {},
    create: {
      title: 'The Future of Artificial Intelligence in Healthcare',
      slug: 'ai-in-healthcare',
      excerpt: 'Explore how AI is revolutionizing medical diagnosis and treatment.',
      cover: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=400&fit=crop',
      content: `Artificial intelligence is transforming the healthcare industry. From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.

One of the most promising applications is in medical imaging analysis. AI algorithms can detect patterns in X-rays, MRIs, and CT scans that might be missed by human eyes.

Another exciting development is in drug discovery. AI can analyze vast databases of molecular structures to identify potential drug candidates much faster than traditional methods.

The future of AI in healthcare looks bright, with potential applications ranging from robotic surgery to virtual health assistants.`,
      status: 'published',
      difficulty: 'intermediate',
      categoryId: categories[0].id,
      authorId: admin.id,
      views: 2340,
      bookmarks: 156,
      ArticleTag: {
        create: [
          { tagId: tags[0].id },
          { tagId: tags[3].id }
        ]
      },
      Sentence: {
        createMany: {
          data: [
            { order: 0, en: 'Artificial intelligence is transforming the healthcare industry.', cn: '人工智能正在改变医疗行业。' },
            { order: 1, en: 'From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.', cn: '从诊断工具到个性化治疗方案，AI正在彻底改变患者护理。' },
            { order: 2, en: 'One of the most promising applications is in medical imaging analysis.', cn: '最有前途的应用之一是医学影像分析。' },
            { order: 3, en: 'AI algorithms can detect patterns in X-rays, MRIs, and CT scans that might be missed by human eyes.', cn: 'AI算法可以检测X射线、MRI和CT扫描中人眼可能遗漏的模式。' }
          ]
        }
      }
    }
  })

  console.log('Created sample article:', article.title)

  // Create sample reading history for test user
  const existingHistory = await prisma.readingHistory.findFirst({
    where: { userId: user.id, articleId: article.id }
  })

  if (!existingHistory) {
    await prisma.readingHistory.create({
      data: {
        userId: user.id,
        articleId: article.id,
        progress: 100,
        lastReadAt: new Date(),
        completedAt: new Date()
      }
    })
    console.log('Created reading history for test user')
  }

  // Create sample bookmarks for test user
  const existingBookmark = await prisma.bookmark.findFirst({
    where: { userId: user.id, articleId: article.id }
  })

  if (!existingBookmark) {
    await prisma.bookmark.create({
      data: {
        userId: user.id,
        articleId: article.id
      }
    })
    console.log('Created bookmark for test user')
  }

  // Create sample vocabulary for test user
  const vocabularyWords = [
    {
      word: 'artificial',
      phonetic: '/ˌɑːrtɪˈfɪʃl/',
      definition: 'Made or produced by human beings rather than occurring naturally',
      example: 'Artificial intelligence is transforming many industries.',
      progress: 80
    },
    {
      word: 'diagnosis',
      phonetic: '/ˌdaɪəɡˈnoʊsɪs/',
      definition: 'The identification of the nature of an illness or problem',
      example: 'Early diagnosis is crucial for effective treatment.',
      progress: 60
    },
    {
      word: 'revolutionize',
      phonetic: '/ˌrevəˈluːʃənaɪz/',
      definition: 'To change something radically or fundamentally',
      example: 'The internet has revolutionized how we communicate.',
      progress: 100
    },
    {
      word: 'unprecedented',
      phonetic: '/ʌnˈpresɪdentɪd/',
      definition: 'Never done or known before',
      example: 'The pandemic caused unprecedented changes in society.',
      progress: 40
    },
    {
      word: 'implement',
      phonetic: '/ˈɪmplɪment/',
      definition: 'To put a decision, plan, or agreement into effect',
      example: 'The company plans to implement new policies next month.',
      progress: 50
    }
  ]

  for (const vocab of vocabularyWords) {
    await prisma.vocabulary.upsert({
      where: {
        userId_word: {
          userId: user.id,
          word: vocab.word
        }
      },
      update: {},
      create: {
        userId: user.id,
        word: vocab.word,
        phonetic: vocab.phonetic,
        definition: vocab.definition,
        example: vocab.example,
        progress: vocab.progress,
        articleId: article.id
      }
    })
  }

  console.log('Created vocabulary for test user')

  console.log('\n✅ Seed completed!')
  console.log('\n📝 Test Accounts:')
  console.log('  Admin: admin@example.com / admin123')
  console.log('  User:  user@example.com / user123')
}

main()
  .catch((e) => {
    console.error(e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
  })