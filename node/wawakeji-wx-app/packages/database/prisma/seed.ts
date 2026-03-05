import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

const sampleArticles = [
  {
    title: 'Getting Started with React Server Components',
    slug: 'react-server-components-intro',
    summary: 'Learn about React Server Components and how they can improve your application performance.',
    content: '# Getting Started with React Server Components\n\nReact Server Components are a new way to build React applications...',
    category: 'frontend',
    difficulty: 'intermediate',
    coverImage: null,
    published: true,
    publishedAt: new Date(),
  },
  {
    title: 'Understanding TypeScript Generics',
    slug: 'typescript-generics-guide',
    summary: 'A comprehensive guide to mastering TypeScript generics with practical examples.',
    content: '# Understanding TypeScript Generics\n\nGenerics are one of the most powerful features in TypeScript...',
    category: 'frontend',
    difficulty: 'advanced',
    coverImage: null,
    published: true,
    publishedAt: new Date(),
  },
  {
    title: 'Introduction to Rust for JavaScript Developers',
    slug: 'rust-for-js-developers',
    summary: 'Learn Rust programming language from a JavaScript perspective.',
    content: '# Introduction to Rust for JavaScript Developers\n\nRust is a systems programming language that has been gaining popularity...',
    category: 'backend',
    difficulty: 'advanced',
    coverImage: null,
    published: true,
    publishedAt: new Date(),
  },
  {
    title: 'Building REST APIs with Node.js',
    slug: 'nodejs-rest-api',
    summary: 'Step by step guide to building RESTful APIs with Node.js and Express.',
    content: '# Building REST APIs with Node.js\n\nNode.js has become one of the most popular choices for building backend services...',
    category: 'backend',
    difficulty: 'beginner',
    coverImage: null,
    published: true,
    publishedAt: new Date(),
  },
  {
    title: 'Docker Best Practices for Developers',
    slug: 'docker-best-practices',
    summary: 'Learn the best practices for using Docker in your development workflow.',
    content: '# Docker Best Practices for Developers\n\nDocker has revolutionized how we develop and deploy applications...',
    category: 'devops',
    difficulty: 'intermediate',
    coverImage: null,
    published: true,
    publishedAt: new Date(),
  },
]

const sampleSentences = [
  {
    content: 'React Server Components are a new way to build React applications.',
    translation: 'React Server Components 是一种构建 React 应用的新方式。',
    ipa: '',
    order: 1,
  },
  {
    content: 'They allow you to render components on the server instead of the client.',
    translation: '它们允许你在服务器端而不是客户端渲染组件。',
    ipa: '',
    order: 2,
  },
  {
    content: 'This can significantly improve performance and reduce bundle size.',
    translation: '这可以显著提高性能并减少打包体积。',
    ipa: '',
    order: 3,
  },
]

async function seed() {
  console.log('Starting database seed...')

  // Create articles
  for (const articleData of sampleArticles) {
    const article = await prisma.article.create({
      data: articleData,
    })
    console.log(`Created article: ${article.title}`)

    // Create sample sentences for each article
    for (const sentence of sampleSentences) {
      await prisma.sentence.create({
        data: {
          ...sentence,
          articleId: article.id,
        },
      })
    }
  }

  console.log('Seed completed successfully!')
}

seed()
  .catch((e) => {
    console.error('Seed error:', e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
  })
