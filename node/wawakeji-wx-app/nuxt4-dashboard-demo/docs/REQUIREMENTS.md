# English Reading App - Requirements Document

## Project Overview

### Project Name
Wawa English Reading App (蛙蛙英语阅读)

### Target Users
Programmers learning English for professional development

### Platform
- **Web Application**: Full-featured reading experience and admin dashboard
- **WeChat Mini Program**: Mobile reading experience with shared user accounts

### Project Goals
1. Help programmers improve English reading skills through immersive article reading
2. Provide sentence-by-sentence audio playback for listening practice
3. Offer AI-powered translation and sentence splitting for better comprehension
4. Enable word lookup for vocabulary building
5. Track reading progress and provide personalized recommendations

---

## User Roles and Permissions

### 1. Admin
**Description**: System administrators who manage content and users

**Permissions**:
- Full access to admin dashboard
- Create, edit, delete articles
- Manage article categories and tags
- Generate AI content (sentence splitting, translation, TTS audio)
- Manage users (view, edit roles, ban/unban)
- View system analytics and statistics
- Configure system settings

### 2. Free User
**Description**: Registered users with basic access

**Permissions**:
- Browse articles by category
- Read articles with sentence-by-sentence view
- Listen to AI-generated audio
- Use word lookup feature
- View reading history (limited to 10 articles)
- Basic reading statistics

### 3. Premium User (Member)
**Description**: Paid subscribers with full access

**Permissions**:
- All Free User permissions, plus:
- Unlimited reading history
- Save/bookmark articles
- Share articles with custom notes
- Save reading recordings
- Advanced reading statistics and progress tracking
- Personalized article recommendations
- Priority access to new features

---

## Feature Specifications

### Module 1: Authentication System

#### 1.1 Email/Password Registration
- User registration with email verification
- Password strength requirements
- Password reset via email

#### 1.2 WeChat Login Integration
- **Web**: WeChat QR code scan login
- **Mini Program**: Direct WeChat authorization
- Account binding: Link WeChat account to existing email account
- Unified user identity across platforms

#### 1.3 User Profile
- Basic info: name, avatar, email
- English level preference
- Interest categories selection
- Membership status and expiration

---

### Module 2: Admin Dashboard

#### 2.1 Article Management
- **Create Article**
  - Title (English)
  - Original content (English text)
  - Category selection
  - Tags management
  - Difficulty level (Beginner/Intermediate/Advanced)
  - Estimated reading time
  - Cover image upload

- **Edit Article**
  - All fields editable
  - Version history tracking

- **Delete Article**
  - Soft delete with confirmation
  - Restore capability for 30 days

- **Article List View**
  - Search by title/content
  - Filter by category, status, date
  - Bulk operations

#### 2.2 AI Content Generation
- **Sentence Splitting**
  - AI-powered automatic sentence detection
  - Manual adjustment capability
  - Sentence order management

- **Translation Generation**
  - AI-powered Chinese translation per sentence
  - Manual correction and approval workflow
  - Translation quality review

- **TTS Audio Generation**
  - Generate audio for each sentence
  - Voice selection (male/female, accent options)
  - Speed adjustment
  - Audio file storage management
  - Regeneration capability

#### 2.3 User Management
- User list with search and filters
- View user details and activity
- Role assignment (Admin/User)
- Ban/unban users
- Membership management

#### 2.4 Category & Tag Management
- Create/edit/delete categories
- Category hierarchy support
- Tag management
- Popular tags tracking

#### 2.5 Analytics Dashboard
- Daily/weekly/monthly active users
- Article view statistics
- Reading completion rates
- Popular articles ranking
- User engagement metrics

---

### Module 3: Reading Experience

#### 3.1 Article Browser
- Category-based navigation
- Search functionality
- Difficulty filter
- Recommended articles section
- Recently viewed articles

#### 3.2 Immersive Reading Interface
- **Sentence-by-Sentence Display**
  - Clean, distraction-free layout
  - Current sentence highlighting
  - Smooth transition between sentences

- **Display Options**
  - Show/hide phonetic transcription (IPA)
  - Show/hide Chinese translation
  - Font size adjustment
  - Theme selection (light/dark/sepia)

- **Audio Player**
  - Play/pause controls
  - Previous/next sentence navigation
  - Speed adjustment (0.5x - 2.0x)
  - Auto-play mode with configurable pause intervals
  - Progress bar with sentence markers

- **Follow-Along Mode**
  - Auto-play full text
  - Pause after each sentence for user to repeat
  - Configurable pause duration
  - Visual countdown indicator

#### 3.3 Word Lookup
- Click/tap on any word to see definition
- Dictionary API integration
- Show:
  - Word pronunciation (phonetics)
  - Part of speech
  - Multiple definitions
  - Example sentences
- Save word to vocabulary list (Premium)

#### 3.4 Reading Progress
- Track reading position
- Resume from last position
- Reading completion percentage
- Time spent reading

---

### Module 4: User Features

#### 4.1 Interest Selection
- Select categories during onboarding
- Modify interests in settings
- Used for personalized recommendations

#### 4.2 Reading History
- List of read articles
- Reading time per article
- Completion status
- Last read timestamp
- Quick resume functionality

#### 4.3 Bookmarks (Premium)
- Save articles for later
- Organize with custom collections
- Add personal notes

#### 4.4 Vocabulary List (Premium)
- Saved words from reading
- Spaced repetition reminders
- Export capability

#### 4.5 Article Sharing (Premium)
- Share article link
- Add personal reading notes
- Share to WeChat moments
- Share card generation

#### 4.6 Reading Statistics
- Total articles read
- Total reading time
- Words learned
- Reading streak
- Weekly/monthly progress charts

---

### Module 5: Membership System

#### 5.1 Subscription Plans
- Monthly subscription
- Quarterly subscription (discounted)
- Annual subscription (best value)

#### 5.2 Payment Integration
- WeChat Pay integration
- Secure payment processing
- Payment history
- Invoice generation

#### 5.3 Membership Features
- Feature gate: check membership status
- Grace period for expired members
- Renewal reminders

---

## Technical Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
├─────────────────────────────┬───────────────────────────────────┤
│      Web Application        │      WeChat Mini Program          │
│   (Nuxt 4 + Nuxt UI v4)     │     (WeChat Mini Program SDK)     │
└──────────────┬──────────────┴──────────────┬────────────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                         API Gateway                               │
│                    (Nuxt Server API)                              │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                       Business Logic Layer                        │
├─────────────┬─────────────┬─────────────┬─────────────┬──────────┤
│   Auth      │  Article    │   User      │   Payment   │   AI     │
│  Service    │  Service    │  Service    │  Service    │ Service  │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                        Data Layer                                 │
├─────────────────────────────┬─────────────────────────────────────┤
│    Prisma ORM               │          File Storage               │
│    (SQLite / MySQL)         │     (Local / S3 Compatible)         │
└─────────────────────────────┴─────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                     External Services                             │
├─────────────┬─────────────┬─────────────┬─────────────┬──────────┤
│   TTS API   │   AI API    │  Dictionary │  WeChat API │ Payment  │
│  (OpenAI)   │  (OpenAI)   │     API     │             │ Gateway  │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend (Web) | Nuxt 4, Vue 3, Tailwind CSS 4, Nuxt UI v4 |
| Frontend (Mini Program) | WeChat Mini Program SDK |
| Backend | Nuxt Server API (Nitro) |
| Database | Prisma ORM + SQLite (dev) / MySQL (prod) |
| File Storage | Local filesystem / S3-compatible storage |
| Authentication | Nuxt Auth Utils, WeChat OAuth |
| TTS Service | OpenAI TTS API / Other TTS providers |
| AI Service | OpenAI GPT API for translation & sentence splitting |
| Dictionary | Free Dictionary API / Custom dictionary |
| Payment | WeChat Pay API |

---

## Database Schema Design

### Prisma Schema

```prisma
// User and Authentication
model User {
  id            String    @id @default(cuid())
  email         String?   @unique
  passwordHash  String?
  name          String?
  avatar        String?
  role          Role      @default(USER)
  status        UserStatus @default(ACTIVE)

  // WeChat integration
  wechatOpenId  String?   @unique
  wechatUnionId String?

  // Membership
  membership    Membership?

  // Preferences
  interests     UserInterest[]
  vocabulary    Vocabulary[]
  bookmarks     Bookmark[]
  readingHistory ReadingHistory[]

  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
}

enum Role {
  ADMIN
  USER
}

enum UserStatus {
  ACTIVE
  BANNED
}

model Membership {
  id            String    @id @default(cuid())
  userId        String    @unique
  user          User      @relation(fields: [userId], references: [id])

  plan          Plan      @default(MONTHLY)
  status        MembershipStatus @default(ACTIVE)
  startDate     DateTime
  endDate       DateTime

  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
}

enum Plan {
  MONTHLY
  QUARTERLY
  ANNUAL
}

enum MembershipStatus {
  ACTIVE
  EXPIRED
  CANCELLED
}

// Categories and Tags
model Category {
  id            String    @id @default(cuid())
  name          String    @unique
  slug          String    @unique
  description   String?
  icon          String?
  parentId      String?
  parent        Category? @relation("CategoryTree", fields: [parentId], references: [id])
  children      Category[] @relation("CategoryTree")
  articles      Article[]
  userInterests UserInterest[]

  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
}

model Tag {
  id            String    @id @default(cuid())
  name          String    @unique
  slug          String    @unique
  articles      Article[]

  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
}

// Article Content
model Article {
  id                String    @id @default(cuid())
  title             String
  content           String      // Original English text
  coverImage        String?

  // Classification
  categoryId        String
  category          Category    @relation(fields: [categoryId], references: [id])
  tags              ArticleTag[]
  difficulty        Difficulty  @default(INTERMEDIATE)
  estimatedTime     Int         // Minutes

  // Status
  status            ArticleStatus @default(DRAFT)
  publishedAt       DateTime?

  // AI Generated Content
  sentences         Sentence[]

  // Statistics
  viewCount         Int         @default(0)
  readCount         Int         @default(0)

  // Relations
  bookmarks         Bookmark[]
  readingHistory    ReadingHistory[]

  createdAt         DateTime    @default(now())
  updatedAt         DateTime    @updatedAt

  @@index([categoryId])
  @@index([status])
}

enum Difficulty {
  BEGINNER
  INTERMEDIATE
  ADVANCED
}

enum ArticleStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
  DELETED
}

model ArticleTag {
  articleId  String
  tagId      String
  article    Article @relation(fields: [articleId], references: [id])
  tag        Tag     @relation(fields: [tagId], references: [id])

  @@id([articleId, tagId])
}

model Sentence {
  id              String    @id @default(cuid())
  articleId       String
  article         Article   @relation(fields: [articleId], references: [id])

  order           Int       // Sentence order in article
  content         String    // English sentence
  translation     String?   // Chinese translation
  phonetic        String?   // IPA phonetic transcription

  // Audio
  audioUrl        String?   // TTS generated audio URL
  audioDuration   Float?    // Audio duration in seconds

  createdAt       DateTime  @default(now())
  updatedAt       DateTime  @updatedAt

  @@unique([articleId, order])
  @@index([articleId])
}

// User Interactions
model UserInterest {
  userId      String
  categoryId  String
  user        User      @relation(fields: [userId], references: [id])
  category    Category  @relation(fields: [categoryId], references: [id])

  createdAt   DateTime  @default(now())

  @@id([userId, categoryId])
}

model Bookmark {
  id          String    @id @default(cuid())
  userId      String
  articleId   String
  user        User      @relation(fields: [userId], references: [id])
  article     Article   @relation(fields: [articleId], references: [id])
  notes       String?

  createdAt   DateTime  @default(now())

  @@unique([userId, articleId])
}

model ReadingHistory {
  id            String    @id @default(cuid())
  userId        String
  articleId     String
  user          User      @relation(fields: [userId], references: [id])
  article       Article   @relation(fields: [articleId], references: [id])

  progress      Float     @default(0)  // Percentage 0-100
  lastPosition  Int       @default(0)  // Sentence index
  timeSpent     Int       @default(0)  // Total seconds
  completed     Boolean   @default(false)

  lastReadAt    DateTime  @default(now())
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  @@unique([userId, articleId])
  @@index([userId])
  @@index([articleId])
}

model Vocabulary {
  id          String    @id @default(cuid())
  userId      String
  user        User      @relation(fields: [userId], references: [id])

  word        String
  definition  String
  context     String?   // Sentence where word was found
  articleId   String?   // Source article

  mastered    Boolean   @default(false)
  reviewCount Int       @default(0)
  lastReviewAt DateTime?

  createdAt   DateTime  @default(now())
  updatedAt   DateTime  @updatedAt

  @@unique([userId, word])
  @@index([userId])
}

// Payment Records
model Payment {
  id              String    @id @default(cuid())
  userId          String
  user            User      @relation(fields: [userId], references: [id])

  amount          Float
  currency        String    @default("CNY")
  plan            Plan

  status          PaymentStatus @default(PENDING)
  transactionId   String?   @unique  // WeChat transaction ID

  paidAt          DateTime?
  createdAt       DateTime  @default(now())
  updatedAt       DateTime  @updatedAt
}

enum PaymentStatus {
  PENDING
  SUCCESS
  FAILED
  REFUNDED
}
```

---

## API Endpoints Design

### Authentication APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login with email/password |
| POST | `/api/auth/logout` | Logout current user |
| POST | `/api/auth/wechat` | WeChat login |
| GET | `/api/auth/me` | Get current user info |
| PUT | `/api/auth/profile` | Update user profile |
| POST | `/api/auth/password/reset` | Request password reset |

### Article APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/articles` | List articles (with filters) |
| GET | `/api/articles/:id` | Get article details |
| GET | `/api/articles/:id/sentences` | Get article sentences |
| POST | `/api/admin/articles` | Create article (Admin) |
| PUT | `/api/admin/articles/:id` | Update article (Admin) |
| DELETE | `/api/admin/articles/:id` | Delete article (Admin) |
| POST | `/api/admin/articles/:id/split` | AI sentence split (Admin) |
| POST | `/api/admin/articles/:id/translate` | AI translation (Admin) |
| POST | `/api/admin/articles/:id/tts` | Generate TTS audio (Admin) |

### Category APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/categories` | List all categories |
| GET | `/api/categories/:slug` | Get category by slug |
| POST | `/api/admin/categories` | Create category (Admin) |
| PUT | `/api/admin/categories/:id` | Update category (Admin) |
| DELETE | `/api/admin/categories/:id` | Delete category (Admin) |

### User Reading APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/reading/history` | Get reading history |
| POST | `/api/reading/progress` | Update reading progress |
| GET | `/api/reading/bookmarks` | Get bookmarks |
| POST | `/api/reading/bookmarks` | Add bookmark |
| DELETE | `/api/reading/bookmarks/:id` | Remove bookmark |
| GET | `/api/reading/vocabulary` | Get vocabulary list |
| POST | `/api/reading/vocabulary` | Add word to vocabulary |
| DELETE | `/api/reading/vocabulary/:id` | Remove from vocabulary |
| GET | `/api/reading/statistics` | Get reading statistics |

### Dictionary API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dictionary/:word` | Look up word definition |

### Payment APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/payment/create` | Create payment order |
| POST | `/api/payment/wechat/notify` | WeChat payment callback |
| GET | `/api/payment/history` | Payment history |
| GET | `/api/membership/status` | Get membership status |

### Admin APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/users` | List users |
| PUT | `/api/admin/users/:id` | Update user |
| GET | `/api/admin/analytics` | Get analytics data |

---

## Third-Party Service Integration

### 1. TTS Service (Text-to-Speech)

**Provider Options**:
- OpenAI TTS API (Primary)
- Azure Cognitive Services
- Google Cloud TTS
- Alibaba Cloud TTS

**Implementation**:
```typescript
// server/utils/tts.ts
interface TTSOptions {
  text: string
  voice?: 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer'
  speed?: number // 0.25 to 4.0
}

export async function generateTTS(options: TTSOptions): Promise<string> {
  // Call TTS API
  // Store audio file
  // Return audio URL
}
```

### 2. AI Service (Sentence Splitting & Translation)

**Provider**: OpenAI GPT API

**Sentence Splitting Prompt**:
```
Split the following English text into individual sentences.
Return a JSON array of sentences, maintaining original punctuation.
Text: {article_content}
```

**Translation Prompt**:
```
Translate the following English sentence to Chinese.
Provide a natural, context-appropriate translation.
Sentence: {sentence}
```

### 3. Dictionary API

**Options**:
- Free Dictionary API (free, limited)
- Oxford Dictionaries API (paid)
- Merriam-Webster API (free tier available)

### 4. WeChat Integration

**Web Login Flow**:
1. Display QR code with unique scene ID
2. User scans with WeChat
3. WeChat sends callback with user info
4. Link or create user account

**Mini Program Login Flow**:
1. Call `wx.login()` to get code
2. Send code to backend
3. Backend exchanges code for session_key and openid
4. Create or link user account

### 5. WeChat Pay Integration

**Payment Flow**:
1. Create order in database
2. Call WeChat Pay unified order API
3. Return payment parameters to client
4. Handle payment result callback
5. Update order status

---

## File Storage Strategy

### Directory Structure

```
uploads/
├── articles/
│   ├── covers/          # Article cover images
│   │   └── {article_id}/
│   │       └── {filename}
│   └── audio/           # TTS generated audio
│       └── {article_id}/
│           └── sentence_{n}.mp3
├── avatars/             # User avatars
│   └── {user_id}/
│       └── {filename}
└── exports/             # User data exports
    └── {user_id}/
        └── vocabulary_{date}.csv
```

### Storage Configuration

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  runtimeConfig: {
    storage: {
      driver: process.env.STORAGE_DRIVER || 'local', // 'local' or 's3'
      local: {
        basePath: process.env.STORAGE_PATH || './uploads'
      },
      s3: {
        endpoint: process.env.S3_ENDPOINT,
        bucket: process.env.S3_BUCKET,
        region: process.env.S3_REGION,
        accessKeyId: process.env.S3_ACCESS_KEY,
        secretAccessKey: process.env.S3_SECRET_KEY
      }
    }
  }
})
```

---

## WeChat Mini Program Integration

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Shared Backend API (Nuxt)                   │
└───────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
┌───────────▼───────────┐   ┌───────────────▼───────────────┐
│   Web Application     │   │    WeChat Mini Program         │
│   (Nuxt 4 SPA)        │   │    (Native Mini Program)       │
└───────────────────────┘   └───────────────────────────────┘
```

### Shared Components
- User authentication (WeChat OAuth)
- Article content API
- Reading progress API
- Payment processing

### Mini Program Specific Features
- Optimized mobile reading UI
- Offline article caching
- WeChat native sharing
- Push notifications for reminders

### Mini Program Pages
1. **Home**: Article recommendations and categories
2. **Article List**: Browse articles by category
3. **Article Detail**: Reading interface
4. **Profile**: User info and membership
5. **History**: Reading history
6. **Bookmarks**: Saved articles (Premium)
7. **Vocabulary**: Word list (Premium)

---

## Security Considerations

### Authentication Security
- Password hashing with bcrypt
- JWT token-based authentication
- Refresh token rotation
- Rate limiting on auth endpoints

### API Security
- Input validation with Zod
- SQL injection prevention (Prisma)
- XSS prevention
- CSRF protection
- Rate limiting per user

### Data Protection
- Sensitive data encryption
- Secure file upload validation
- Content Security Policy
- HTTPS enforcement

### Payment Security
- Signature verification for callbacks
- Idempotency for payment operations
- Secure credential storage

---

## Performance Requirements

### Response Time
- API response: < 200ms (p95)
- Page load: < 2s (initial)
- Audio streaming: Start within 500ms

### Scalability
- Support 10,000 concurrent users
- Handle 100 requests/second
- Database query optimization with proper indexing

### Caching Strategy
- Redis for session storage
- In-memory cache for frequently accessed articles
- CDN for static assets and audio files

---

## Development Methodology

### Test-Driven Development (TDD)

#### Backend Testing
- Unit tests for all services
- Integration tests for API endpoints
- Database tests with test database

#### Frontend Testing
- Component tests with Vitest
- E2E tests with Playwright
- Visual regression tests (optional)

#### Test Structure
```
tests/
├── unit/
│   ├── server/
│   │   ├── services/
│   │   └── utils/
│   └── app/
│       └── components/
├── integration/
│   └── api/
└── e2e/
    └── specs/
```

### Code Quality
- ESLint configuration
- TypeScript strict mode
- Pre-commit hooks with lint-staged
- Code review requirements

---

## Deployment Strategy

### Development Environment
- Local SQLite database
- Local file storage
- Environment variables in `.env`

### Production Environment
- MySQL database (managed service)
- S3-compatible object storage
- Environment variables in platform settings
- Docker containerization (optional)

### CI/CD Pipeline
1. Run tests on PR
2. Build and deploy on merge to main
3. Database migrations
4. Static asset deployment

---

## Future Enhancements

### Phase 2 Features
- Reading comprehension quizzes
- Speaking practice with speech recognition
- Community features (comments, discussions)
- Leaderboard and achievements
- Spaced repetition vocabulary practice

### Phase 3 Features
- AI-generated articles
- Custom reading plans
- Team/Organization accounts
- API for third-party integrations

---

## Glossary

| Term | Definition |
|------|-----------|
| TTS | Text-to-Speech, technology that converts text to spoken audio |
| IPA | International Phonetic Alphabet, notation for pronunciation |
| OAuth | Open Authorization, standard for access delegation |
| JWT | JSON Web Token, compact token format for authentication |
| Prisma | Next-generation ORM for Node.js and TypeScript |
| Nuxt | Vue.js meta-framework for building full-stack applications |