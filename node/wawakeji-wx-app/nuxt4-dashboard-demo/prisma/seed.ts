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
      splitContent: {
        paragraphs: [
          {
            order: 0,
            en: 'Artificial intelligence is transforming the healthcare industry. From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.',
            cn: '人工智能正在改变医疗行业。从诊断工具到个性化治疗方案，AI正在彻底改变患者护理。',
            audio: 'tts://placeholder/paragraph/0'
          },
          {
            order: 1,
            en: 'One of the most promising applications is in medical imaging analysis. AI algorithms can detect patterns in X-rays, MRIs, and CT scans that might be missed by human eyes.',
            cn: '最有前途的应用之一是医学影像分析。AI算法可以检测X射线、MRI和CT扫描中人眼可能遗漏的模式。',
            audio: 'tts://placeholder/paragraph/1'
          },
          {
            order: 2,
            en: 'Another exciting development is in drug discovery. AI can analyze vast databases of molecular structures to identify potential drug candidates much faster than traditional methods.',
            cn: '另一个令人兴奋的发展是药物发现。AI可以分析大量分子结构数据库，比传统方法更快地识别潜在的候选药物。',
            audio: 'tts://placeholder/paragraph/2'
          },
          {
            order: 3,
            en: 'The future of AI in healthcare looks bright, with potential applications ranging from robotic surgery to virtual health assistants.',
            cn: '人工智能在医疗保健领域的未来看起来很光明，潜在应用范围从机器人手术到虚拟健康助手。',
            audio: 'tts://placeholder/paragraph/3'
          }
        ],
        sentences: [
          { order: 0, paragraphIndex: 0, en: 'Artificial intelligence is transforming the healthcare industry.', cn: '人工智能正在改变医疗行业。', audio: 'tts://placeholder/sentence/0', phonetics: [{ word: 'artificial', phonetic: '/ˌɑːrtɪˈfɪʃl/' }, { word: 'intelligence', phonetic: '/ɪnˈtelɪdʒəns/' }, { word: 'transforming', phonetic: '/trænsˈfɔːrmɪŋ/' }, { word: 'healthcare', phonetic: '/ˈhelθkeər/' }] },
          { order: 1, paragraphIndex: 0, en: 'From diagnostic tools to personalized treatment plans, AI is revolutionizing patient care.', cn: '从诊断工具到个性化治疗方案，AI正在彻底改变患者护理。', audio: 'tts://placeholder/sentence/1', phonetics: [{ word: 'diagnostic', phonetic: '/ˌdaɪəɡˈnɑːstɪk/' }, { word: 'personalized', phonetic: '/ˈpɜːrsənəlaɪzd/' }, { word: 'treatment', phonetic: '/ˈtriːtmənt/' }, { word: 'revolutionizing', phonetic: '/ˌrevəˈluːʃənaɪzɪŋ/' }] },
          { order: 2, paragraphIndex: 1, en: 'One of the most promising applications is in medical imaging analysis.', cn: '最有前途的应用之一是医学影像分析。', audio: 'tts://placeholder/sentence/2', phonetics: [{ word: 'promising', phonetic: '/ˈprɑːmɪsɪŋ/' }, { word: 'applications', phonetic: '/ˌæplɪˈkeɪʃənz/' }, { word: 'imaging', phonetic: '/ˈɪmɪdʒɪŋ/' }, { word: 'analysis', phonetic: '/əˈnæləsɪs/' }] },
          { order: 3, paragraphIndex: 1, en: 'AI algorithms can detect patterns in X-rays, MRIs, and CT scans that might be missed by human eyes.', cn: 'AI算法可以检测X射线、MRI和CT扫描中人眼可能遗漏的模式。', audio: 'tts://placeholder/sentence/3', phonetics: [{ word: 'algorithms', phonetic: '/ˈælɡərɪðəmz/' }, { word: 'detect', phonetic: '/dɪˈtekt/' }, { word: 'patterns', phonetic: '/ˈpætərnz/' }] },
          { order: 4, paragraphIndex: 2, en: 'Another exciting development is in drug discovery.', cn: '另一个令人兴奋的发展是药物发现。', audio: 'tts://placeholder/sentence/4', phonetics: [{ word: 'exciting', phonetic: '/ɪkˈsaɪtɪŋ/' }, { word: 'development', phonetic: '/dɪˈveləpmənt/' }, { word: 'discovery', phonetic: '/dɪˈskʌvəri/' }] },
          { order: 5, paragraphIndex: 2, en: 'AI can analyze vast databases of molecular structures to identify potential drug candidates much faster than traditional methods.', cn: 'AI可以分析大量分子结构数据库，比传统方法更快地识别潜在的候选药物。', audio: 'tts://placeholder/sentence/5', phonetics: [{ word: 'analyze', phonetic: '/ˈænəlaɪz/' }, { word: 'molecular', phonetic: '/məˈlekjələr/' }, { word: 'structures', phonetic: '/ˈstrʌktʃərz/' }, { word: 'candidates', phonetic: '/ˈkændɪdeɪts/' }] },
          { order: 6, paragraphIndex: 3, en: 'The future of AI in healthcare looks bright, with potential applications ranging from robotic surgery to virtual health assistants.', cn: '人工智能在医疗保健领域的未来看起来很光明，潜在应用范围从机器人手术到虚拟健康助手。', audio: 'tts://placeholder/sentence/6', phonetics: [{ word: 'potential', phonetic: '/pəˈtenʃl/' }, { word: 'applications', phonetic: '/ˌæplɪˈkeɪʃənz/' }, { word: 'robotic', phonetic: '/roʊˈbɑːtɪk/' }, { word: 'surgery', phonetic: '/ˈsɜːrdʒəri/' }, { word: 'virtual', phonetic: '/ˈvɜːrtʃuəl/' }] }
        ]
      },
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
      splitContent: {
        paragraphs: [
          {
            order: 0,
            en: 'Climate change remains one of the most pressing issues of our time. Scientists around the world are conducting extensive research to understand its causes and effects.',
            cn: '气候变化仍然是我们这个时代最紧迫的问题之一。世界各地的科学家正在进行广泛的研究，以了解其原因和影响。',
            audio: 'tts://placeholder/paragraph/0'
          },
          {
            order: 1,
            en: 'Recent studies show that global temperatures have risen by approximately 1.1 degrees Celsius since the pre-industrial era. This may seem small, but the consequences are significant.',
            cn: '最近的研究表明，自工业时代以来，全球气温已上升约1.1摄氏度。这看起来可能很小，但后果是显著的。',
            audio: 'tts://placeholder/paragraph/1'
          },
          {
            order: 2,
            en: 'Extreme weather events are becoming more frequent. Heatwaves, droughts, and intense storms are affecting communities worldwide.',
            cn: '极端天气事件正变得更加频繁。热浪、干旱和强烈的风暴正在影响世界各地的社区。',
            audio: 'tts://placeholder/paragraph/2'
          },
          {
            order: 3,
            en: 'The good news is that renewable energy technologies are advancing rapidly. Solar and wind power are now cheaper than fossil fuels in many regions.',
            cn: '好消息是可再生能源技术正在快速进步。太阳能和风能在许多地区现在比化石燃料更便宜。',
            audio: 'tts://placeholder/paragraph/3'
          }
        ],
        sentences: [
          { order: 0, paragraphIndex: 0, en: 'Climate change remains one of the most pressing issues of our time.', cn: '气候变化仍然是我们这个时代最紧迫的问题之一。', audio: 'tts://placeholder/sentence/0', phonetics: [{ word: 'climate', phonetic: '/ˈklaɪmət/' }, { word: 'pressing', phonetic: '/ˈpresɪŋ/' }] },
          { order: 1, paragraphIndex: 0, en: 'Scientists around the world are conducting extensive research to understand its causes and effects.', cn: '世界各地的科学家正在进行广泛的研究，以了解其原因和影响。', audio: 'tts://placeholder/sentence/1', phonetics: [{ word: 'scientists', phonetic: '/ˈsaɪəntɪsts/' }, { word: 'extensive', phonetic: '/ɪkˈstensɪv/' }] },
          { order: 2, paragraphIndex: 1, en: 'Recent studies show that global temperatures have risen by approximately 1.1 degrees Celsius since the pre-industrial era.', cn: '最近的研究表明，自工业时代以来，全球气温已上升约1.1摄氏度。', audio: 'tts://placeholder/sentence/2', phonetics: [{ word: 'approximately', phonetic: '/əˈprɑːksɪmətli/' }, { word: 'celsius', phonetic: '/ˈselsiəs/' }] },
          { order: 3, paragraphIndex: 1, en: 'This may seem small, but the consequences are significant.', cn: '这看起来可能很小，但后果是显著的。', audio: 'tts://placeholder/sentence/3', phonetics: [{ word: 'consequences', phonetic: '/ˈkɑːnsɪkwensɪz/' }, { word: 'significant', phonetic: '/sɪɡˈnɪfɪkənt/' }] },
          { order: 4, paragraphIndex: 2, en: 'Extreme weather events are becoming more frequent.', cn: '极端天气事件正变得更加频繁。', audio: 'tts://placeholder/sentence/4', phonetics: [{ word: 'extreme', phonetic: '/ɪkˈstriːm/' }, { word: 'frequent', phonetic: '/ˈfriːkwənt/' }] },
          { order: 5, paragraphIndex: 2, en: 'Heatwaves, droughts, and intense storms are affecting communities worldwide.', cn: '热浪、干旱和强烈的风暴正在影响世界各地的社区。', audio: 'tts://placeholder/sentence/5', phonetics: [{ word: 'heatwaves', phonetic: '/ˈhiːtweɪvz/' }, { word: 'droughts', phonetic: '/draʊts/' }, { word: 'intense', phonetic: '/ɪnˈtens/' }] },
          { order: 6, paragraphIndex: 3, en: 'The good news is that renewable energy technologies are advancing rapidly.', cn: '好消息是可再生能源技术正在快速进步。', audio: 'tts://placeholder/sentence/6', phonetics: [{ word: 'renewable', phonetic: '/rɪˈnuːəbl/' }, { word: 'technologies', phonetic: '/tekˈnɑːlədʒiz/' }] },
          { order: 7, paragraphIndex: 3, en: 'Solar and wind power are now cheaper than fossil fuels in many regions.', cn: '太阳能和风能在许多地区现在比化石燃料更便宜。', audio: 'tts://placeholder/sentence/7', phonetics: [{ word: 'solar', phonetic: '/ˈsoʊlər/' }, { word: 'fossil', phonetic: '/ˈfɑːsl/' }] }
        ]
      },
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
      splitContent: {
        paragraphs: [
          {
            order: 0,
            en: 'Starting a business is never easy, but learning from successful founders can help you avoid common mistakes.',
            cn: '创业从来都不容易，但从成功的创始人那里学习可以帮助你避免常见的错误。',
            audio: 'tts://placeholder/paragraph/0'
          },
          {
            order: 1,
            en: 'One key lesson is the importance of solving a real problem. The most successful startups address genuine pain points that people experience daily.',
            cn: '一个关键的教训是解决真正问题的重要性。最成功的初创公司解决人们每天经历的真实痛点。',
            audio: 'tts://placeholder/paragraph/1'
          },
          {
            order: 2,
            en: 'Another crucial factor is timing. Many great ideas fail because they enter the market too early or too late.',
            cn: '另一个关键因素是时机。许多好主意因为进入市场太早或太晚而失败。',
            audio: 'tts://placeholder/paragraph/2'
          },
          {
            order: 3,
            en: 'Building a strong team is equally important. You need people with complementary skills who share your vision and values.',
            cn: '建立一支强大的团队同样重要。你需要具有互补技能、分享你愿景和价值观的人。',
            audio: 'tts://placeholder/paragraph/3'
          },
          {
            order: 4,
            en: 'Finally, perseverance is essential. Most successful founders faced numerous rejections before achieving their goals.',
            cn: '最后，坚持不懈至关重要。大多数成功的创始人在实现目标之前都面临过无数次拒绝。',
            audio: 'tts://placeholder/paragraph/4'
          }
        ],
        sentences: [
          { order: 0, paragraphIndex: 0, en: 'Starting a business is never easy, but learning from successful founders can help you avoid common mistakes.', cn: '创业从来都不容易，但从成功的创始人那里学习可以帮助你避免常见的错误。', audio: 'tts://placeholder/sentence/0', phonetics: [{ word: 'business', phonetic: '/ˈbɪznəs/' }] },
          { order: 1, paragraphIndex: 1, en: 'One key lesson is the importance of solving a real problem.', cn: '一个关键的教训是解决真正问题的重要性。', audio: 'tts://placeholder/sentence/1', phonetics: [{ word: 'importance', phonetic: '/ɪmˈpɔːrtəns/' }] },
          { order: 2, paragraphIndex: 1, en: 'The most successful startups address genuine pain points that people experience daily.', cn: '最成功的初创公司解决人们每天经历的真实痛点。', audio: 'tts://placeholder/sentence/2', phonetics: [{ word: 'startups', phonetic: '/ˈstɑːrtʌps/' }, { word: 'genuine', phonetic: '/ˈdʒenjuɪn/' }] },
          { order: 3, paragraphIndex: 2, en: 'Another crucial factor is timing.', cn: '另一个关键因素是时机。', audio: 'tts://placeholder/sentence/3', phonetics: [{ word: 'crucial', phonetic: '/ˈkruːʃl/' }] },
          { order: 4, paragraphIndex: 2, en: 'Many great ideas fail because they enter the market too early or too late.', cn: '许多好主意因为进入市场太早或太晚而失败。', audio: 'tts://placeholder/sentence/4', phonetics: [] },
          { order: 5, paragraphIndex: 3, en: 'Building a strong team is equally important.', cn: '建立一支强大的团队同样重要。', audio: 'tts://placeholder/sentence/5', phonetics: [{ word: 'equally', phonetic: '/ˈiːkwəli/' }] },
          { order: 6, paragraphIndex: 3, en: 'You need people with complementary skills who share your vision and values.', cn: '你需要具有互补技能、分享你愿景和价值观的人。', audio: 'tts://placeholder/sentence/6', phonetics: [{ word: 'complementary', phonetic: '/ˌkɑːmplɪˈmentri/' }] },
          { order: 7, paragraphIndex: 4, en: 'Finally, perseverance is essential.', cn: '最后，坚持不懈至关重要。', audio: 'tts://placeholder/sentence/7', phonetics: [{ word: 'perseverance', phonetic: '/ˌpɜːrsəˈvɪrəns/' }, { word: 'essential', phonetic: '/ɪˈsenʃl/' }] },
          { order: 8, paragraphIndex: 4, en: 'Most successful founders faced numerous rejections before achieving their goals.', cn: '大多数成功的创始人在实现目标之前都面临过无数次拒绝。', audio: 'tts://placeholder/sentence/8', phonetics: [{ word: 'numerous', phonetic: '/ˈnuːmərəs/' }, { word: 'rejections', phonetic: '/rɪˈdʒekʃənz/' }] }
        ]
      },
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
      splitContent: {
        paragraphs: [
          {
            order: 0,
            en: 'Sleep is essential for our physical and mental well-being. Yet in our busy modern lives, many people neglect this fundamental need.',
            cn: '睡眠对我们的身心健康至关重要。然而在我们忙碌的现代生活中，许多人忽视了这一基本需求。',
            audio: 'tts://placeholder/paragraph/0'
          },
          {
            order: 1,
            en: 'During sleep, our bodies repair tissues and consolidate memories. The brain processes information from the day and prepares for tomorrow.',
            cn: '在睡眠期间，我们的身体修复组织并巩固记忆。大脑处理白天的信息并为明天做准备。',
            audio: 'tts://placeholder/paragraph/1'
          },
          {
            order: 2,
            en: 'Lack of sleep has been linked to numerous health problems, including obesity, heart disease, and weakened immunity.',
            cn: '睡眠不足与许多健康问题有关，包括肥胖、心脏病和免疫力下降。',
            audio: 'tts://placeholder/paragraph/2'
          },
          {
            order: 3,
            en: 'Most adults need between 7 and 9 hours of sleep per night. However, quality matters as much as quantity.',
            cn: '大多数成年人每晚需要7到9小时的睡眠。然而，质量与数量同样重要。',
            audio: 'tts://placeholder/paragraph/3'
          },
          {
            order: 4,
            en: 'Creating a consistent sleep schedule and a relaxing bedtime routine can significantly improve your sleep quality.',
            cn: '建立一致的睡眠时间表和放松的睡前习惯可以显著改善你的睡眠质量。',
            audio: 'tts://placeholder/paragraph/4'
          }
        ],
        sentences: [
          { order: 0, paragraphIndex: 0, en: 'Sleep is essential for our physical and mental well-being.', cn: '睡眠对我们的身心健康至关重要。', audio: 'tts://placeholder/sentence/0', phonetics: [{ word: 'essential', phonetic: '/ɪˈsenʃl/' }, { word: 'well-being', phonetic: '/ˈwelbiːɪŋ/' }] },
          { order: 1, paragraphIndex: 0, en: 'Yet in our busy modern lives, many people neglect this fundamental need.', cn: '然而在我们忙碌的现代生活中，许多人忽视了这一基本需求。', audio: 'tts://placeholder/sentence/1', phonetics: [{ word: 'neglect', phonetic: '/nɪˈɡlekt/' }, { word: 'fundamental', phonetic: '/ˌfʌndəˈmentl/' }] },
          { order: 2, paragraphIndex: 1, en: 'During sleep, our bodies repair tissues and consolidate memories.', cn: '在睡眠期间，我们的身体修复组织并巩固记忆。', audio: 'tts://placeholder/sentence/2', phonetics: [{ word: 'tissues', phonetic: '/ˈtɪʃuːz/' }, { word: 'consolidate', phonetic: '/kənˈsɑːlɪdeɪt/' }] },
          { order: 3, paragraphIndex: 1, en: 'The brain processes information from the day and prepares for tomorrow.', cn: '大脑处理白天的信息并为明天做准备。', audio: 'tts://placeholder/sentence/3', phonetics: [{ word: 'processes', phonetic: '/ˈprɑːsesɪz/' }] },
          { order: 4, paragraphIndex: 2, en: 'Lack of sleep has been linked to numerous health problems, including obesity, heart disease, and weakened immunity.', cn: '睡眠不足与许多健康问题有关，包括肥胖、心脏病和免疫力下降。', audio: 'tts://placeholder/sentence/4', phonetics: [{ word: 'obesity', phonetic: '/oʊˈbiːsəti/' }, { word: 'immunity', phonetic: '/ɪˈmjuːnəti/' }] },
          { order: 5, paragraphIndex: 3, en: 'Most adults need between 7 and 9 hours of sleep per night.', cn: '大多数成年人每晚需要7到9小时的睡眠。', audio: 'tts://placeholder/sentence/5', phonetics: [] },
          { order: 6, paragraphIndex: 3, en: 'However, quality matters as much as quantity.', cn: '然而，质量与数量同样重要。', audio: 'tts://placeholder/sentence/6', phonetics: [{ word: 'quantity', phonetic: '/ˈkwɑːntəti/' }] },
          { order: 7, paragraphIndex: 4, en: 'Creating a consistent sleep schedule and a relaxing bedtime routine can significantly improve your sleep quality.', cn: '建立一致的睡眠时间表和放松的睡前习惯可以显著改善你的睡眠质量。', audio: 'tts://placeholder/sentence/7', phonetics: [{ word: 'consistent', phonetic: '/kənˈsɪstənt/' }, { word: 'significantly', phonetic: '/sɪɡˈnɪfɪkəntli/' }] }
        ]
      },
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