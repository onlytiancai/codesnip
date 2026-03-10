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

  // Create sample article (Technology)
  const techArticle = await prisma.article.upsert({
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

  console.log('Created technology article:', techArticle.title)

  // Create Science article
  const scienceArticle = await prisma.article.upsert({
    where: { slug: 'climate-change-research' },
    update: {},
    create: {
      title: 'Climate Change: What Scientists Are Saying',
      slug: 'climate-change-research',
      excerpt: 'Understanding the latest research on global warming and its impacts on our planet.',
      cover: 'https://images.unsplash.com/photo-1569163139599-0f4517e36f51?w=800&h=400&fit=crop',
      content: `Climate change remains one of the most pressing issues of our time. Scientists around the world are conducting extensive research to understand its causes and effects.

Recent studies show that global temperatures have risen by approximately 1.1 degrees Celsius since the pre-industrial era. This may seem small, but the consequences are significant.

Extreme weather events are becoming more frequent. Heatwaves, droughts, and intense storms are affecting communities worldwide.

The good news is that renewable energy technologies are advancing rapidly. Solar and wind power are now cheaper than fossil fuels in many regions.`,
      status: 'published',
      difficulty: 'advanced',
      categoryId: categories[1].id,
      authorId: admin.id,
      views: 1856,
      bookmarks: 98,
      ArticleTag: {
        create: [
          { tagId: tags[4].id }
        ]
      },
      Sentence: {
        createMany: {
          data: [
            { order: 0, en: 'Climate change remains one of the most pressing issues of our time.', cn: '气候变化仍然是我们这个时代最紧迫的问题之一。' },
            { order: 1, en: 'Scientists around the world are conducting extensive research to understand its causes and effects.', cn: '世界各地的科学家正在进行广泛的研究，以了解其原因和影响。' },
            { order: 2, en: 'Recent studies show that global temperatures have risen by approximately 1.1 degrees Celsius since the pre-industrial era.', cn: '最近的研究表明，自工业时代以来，全球气温已上升约1.1摄氏度。' },
            { order: 3, en: 'Extreme weather events are becoming more frequent.', cn: '极端天气事件正变得更加频繁。' }
          ]
        }
      }
    }
  })

  console.log('Created science article:', scienceArticle.title)

  // Create Business article
  const businessArticle = await prisma.article.upsert({
    where: { slug: 'startup-success-stories' },
    update: {},
    create: {
      title: 'Building a Successful Startup: Lessons from Founders',
      slug: 'startup-success-stories',
      excerpt: 'Key insights from entrepreneurs who built billion-dollar companies.',
      cover: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=800&h=400&fit=crop',
      content: `Starting a business is never easy, but learning from successful founders can help you avoid common mistakes.

One key lesson is the importance of solving a real problem. The most successful startups address genuine pain points that people experience daily.

Another crucial factor is timing. Many great ideas fail because they enter the market too early or too late.

Building a strong team is equally important. You need people with complementary skills who share your vision and values.

Finally, perseverance is essential. Most successful founders faced numerous rejections before achieving their goals.`,
      status: 'published',
      difficulty: 'beginner',
      categoryId: categories[2].id,
      authorId: admin.id,
      views: 3127,
      bookmarks: 234,
      ArticleTag: {
        create: [
          { tagId: tags[2].id }
        ]
      },
      Sentence: {
        createMany: {
          data: [
            { order: 0, en: 'Starting a business is never easy, but learning from successful founders can help you avoid common mistakes.', cn: '创业从来都不容易，但从成功的创始人那里学习可以帮助你避免常见的错误。' },
            { order: 1, en: 'One key lesson is the importance of solving a real problem.', cn: '一个关键的教训是解决真正问题的重要性。' },
            { order: 2, en: 'Another crucial factor is timing.', cn: '另一个关键因素是时机。' },
            { order: 3, en: 'Building a strong team is equally important.', cn: '建立一支强大的团队同样重要。' }
          ]
        }
      }
    }
  })

  console.log('Created business article:', businessArticle.title)

  // Create Health article
  const healthArticle = await prisma.article.upsert({
    where: { slug: 'science-of-sleep' },
    update: {},
    create: {
      title: 'The Science of Sleep: Why It Matters',
      slug: 'science-of-sleep',
      excerpt: 'Discover how quality sleep affects your health and productivity.',
      cover: 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=800&h=400&fit=crop',
      content: `Sleep is essential for our physical and mental well-being. Yet in our busy modern lives, many people neglect this fundamental need.

During sleep, our bodies repair tissues and consolidate memories. The brain processes information from the day and prepares for tomorrow.

Lack of sleep has been linked to numerous health problems, including obesity, heart disease, and weakened immunity.

Most adults need between 7 and 9 hours of sleep per night. However, quality matters as much as quantity.

Creating a consistent sleep schedule and a relaxing bedtime routine can significantly improve your sleep quality.`,
      status: 'published',
      difficulty: 'beginner',
      categoryId: categories[3].id,
      authorId: admin.id,
      views: 4231,
      bookmarks: 312,
      ArticleTag: {
        create: [
          { tagId: tags[3].id }
        ]
      },
      Sentence: {
        createMany: {
          data: [
            { order: 0, en: 'Sleep is essential for our physical and mental well-being.', cn: '睡眠对我们的身心健康至关重要。' },
            { order: 1, en: 'During sleep, our bodies repair tissues and consolidate memories.', cn: '在睡眠期间，我们的身体修复组织并巩固记忆。' },
            { order: 2, en: 'Lack of sleep has been linked to numerous health problems.', cn: '睡眠不足与许多健康问题有关。' },
            { order: 3, en: 'Most adults need between 7 and 9 hours of sleep per night.', cn: '大多数成年人每晚需要7到9小时的睡眠。' }
          ]
        }
      }
    }
  })

  console.log('Created health article:', healthArticle.title)

  // Create sample reading history for test user
  const existingHistory = await prisma.readingHistory.findFirst({
    where: { userId: user.id, articleId: techArticle.id }
  })

  if (!existingHistory) {
    await prisma.readingHistory.create({
      data: {
        userId: user.id,
        articleId: techArticle.id,
        progress: 100,
        lastReadAt: new Date(),
        completedAt: new Date()
      }
    })
    console.log('Created reading history for test user')
  }

  // Create sample bookmarks for test user
  const existingBookmark = await prisma.bookmark.findFirst({
    where: { userId: user.id, articleId: techArticle.id }
  })

  if (!existingBookmark) {
    await prisma.bookmark.create({
      data: {
        userId: user.id,
        articleId: techArticle.id
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
        articleId: techArticle.id
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