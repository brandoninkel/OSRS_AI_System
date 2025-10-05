# OSRS AI Assistant - Frontend Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Running the Application](#running-the-application)
7. [Core Components](#core-components)
8. [Features](#features)
9. [Styling System](#styling-system)
10. [State Management](#state-management)
11. [API Integration](#api-integration)
12. [PWA Features](#pwa-features)
13. [Configuration](#configuration)
14. [Development](#development)
15. [Build & Deployment](#build--deployment)

---

## Overview

The OSRS AI Assistant frontend is a modern, responsive web application built with React and Vite. It provides an intuitive chat interface for interacting with the OSRS Agentic RAG system, featuring:

- **Real-time Chat Interface**: Ask questions and get AI-powered answers
- **Attribution System**: See sources and wiki contributors for answers
- **Interactive Tooltips**: Hover over highlighted text to see source details
- **Economic Dashboard**: Track Grand Exchange prices and trends
- **Progressive Web App**: Install as a standalone app on any device
- **Persistent Chat History**: Messages saved to localStorage
- **Responsive Design**: Works on desktop, tablet, and mobile

**Live URL**: http://localhost:3005
**API Endpoint**: http://localhost:5001

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser (User)                          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    React Application                         │
│                      (App.jsx)                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Chat Interface                                       │  │
│  │  - Message List (ScrollArea)                         │  │
│  │  - Input Field                                       │  │
│  │  - Send Button                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Attribution System                                   │  │
│  │  - Show/Hide Attributions Button                     │  │
│  │  - Highlighted Text with Tooltips                    │  │
│  │  - Contributor Information                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  UI Components (Radix UI + Tailwind)                 │  │
│  │  - Card, Button, Input, Badge, etc.                  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                    Flask API Server                          │
│                  http://localhost:5001                       │
│  - POST /chat (send query)                                  │
│  - POST /attributions (get contributors)                    │
│  - GET /health (check status)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Framework
- **React 19.1.1** - UI library
- **Vite 7.1.6** - Build tool and dev server
- **JavaScript (JSX)** - Primary language

### UI Components
- **Radix UI** - Headless, accessible component primitives:
  - `@radix-ui/react-avatar` - Avatar component
  - `@radix-ui/react-checkbox` - Checkbox component
  - `@radix-ui/react-dialog` - Modal dialogs
  - `@radix-ui/react-progress` - Progress bars
  - `@radix-ui/react-scroll-area` - Custom scrollbars
  - `@radix-ui/react-separator` - Dividers
  - `@radix-ui/react-slot` - Component composition
  - `@radix-ui/react-tabs` - Tab navigation
  - `@radix-ui/react-toast` - Toast notifications
  - `@radix-ui/react-tooltip` - Tooltips

### Styling
- **Tailwind CSS 3.4.17** - Utility-first CSS framework
- **tailwindcss-animate** - Animation utilities
- **PostCSS** - CSS processing
- **Autoprefixer** - Vendor prefix automation

### Utilities
- **class-variance-authority** - Component variant management
- **clsx** - Conditional className utility
- **tailwind-merge** - Merge Tailwind classes intelligently
- **lucide-react** - Icon library

### PWA
- **vite-plugin-pwa** - Progressive Web App support
- **Workbox** - Service worker management

### Development Tools
- **ESLint** - Code linting
- **TypeScript** - Type checking (for .tsx files)

---

## Project Structure

```
frontend/
├── public/                      # Static assets
│   ├── manifest.json           # PWA manifest
│   └── vite.svg                # App icon
├── src/                        # Source code
│   ├── components/             # React components
│   │   ├── EconomicDashboard.jsx  # Economic analysis dashboard
│   │   └── ui/                 # Reusable UI components
│   │       ├── avatar.tsx      # Avatar component
│   │       ├── badge.tsx       # Badge component
│   │       ├── button.tsx      # Button component
│   │       ├── card.tsx        # Card component
│   │       ├── checkbox.tsx    # Checkbox component
│   │       ├── dialog.tsx      # Dialog/Modal component
│   │       ├── input.tsx       # Input field component
│   │       ├── progress.tsx    # Progress bar component
│   │       ├── scroll-area.tsx # Scrollable area component
│   │       ├── separator.tsx   # Separator/Divider component
│   │       ├── skeleton.tsx    # Loading skeleton component
│   │       ├── tabs.tsx        # Tab navigation component
│   │       ├── textarea.tsx    # Textarea component
│   │       ├── toast.tsx       # Toast notification component
│   │       ├── toaster.tsx     # Toast container component
│   │       └── tooltip.tsx     # Tooltip component
│   ├── hooks/                  # Custom React hooks
│   │   └── use-toast.ts        # Toast notification hook
│   ├── lib/                    # Utility libraries
│   │   └── utils.ts            # Utility functions (cn)
│   ├── assets/                 # Images and static files
│   │   └── react.svg           # React logo
│   ├── App.jsx                 # Main application component
│   ├── App.css                 # Component-specific styles
│   ├── main.jsx                # Application entry point
│   └── index.css               # Global styles and Tailwind
├── index.html                  # HTML template
├── package.json                # Dependencies and scripts
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
├── postcss.config.js           # PostCSS configuration
├── eslint.config.js            # ESLint configuration
├── tsconfig.json               # TypeScript configuration
└── components.json             # shadcn/ui configuration
```

---

## Installation & Setup

### Prerequisites
- **Node.js 18+** installed
- **npm** or **yarn** package manager
- **API server** running on port 5001

### Install Dependencies
```bash
cd /Users/brandon/Documents/projects/GE/frontend
npm install
```

### Environment Variables
Create a `.env` file (optional):
```bash
VITE_API_BASE=http://localhost:5001
```

If not set, defaults to `http://localhost:5001`.

---

## Running the Application

### Development Mode
```bash
npm run dev
```

**Access**: http://localhost:3005

Features:
- Hot Module Replacement (HMR)
- Fast refresh
- Source maps
- Dev server with CORS

### ⚠️ Troubleshooting: "Offline" Status

If the GUI shows **"Offline"** in the top right:

**1. Check API Server is Running**:
```bash
curl http://localhost:5001/health
```
Should return: `{"status": "healthy"}`

**2. Verify Environment Configuration**:
Check `frontend/.env` file:
```bash
# For local development (default)
VITE_API_BASE=http://localhost:5001

# For network access (comment out for local dev)
# VITE_API_BASE=http://192.168.0.151:5001
```

**3. Refresh Browser**:
- Hard refresh: `Cmd + Shift + R` (macOS) or `Ctrl + Shift + R` (Windows/Linux)
- Or close tab and reopen http://localhost:3005

**4. Check Browser Console**:
- Open DevTools (F12)
- Look for connection errors
- Verify API base URL is correct

### Production Preview
```bash
npm run build
npm run preview
```

### Linting
```bash
npm run lint
```

---

## Core Components

### 1. App.jsx (Main Application)
**Purpose**: Root component containing the entire chat interface

**Key Features**:
- Chat message list with auto-scroll
- Input field for user queries
- API health check indicator
- Attribution system integration
- LocalStorage persistence

**State Management**:
```javascript
const [messages, setMessages] = useState([])        // Chat messages
const [input, setInput] = useState('')              // Input field value
const [loading, setLoading] = useState(false)       // Loading state
const [showAttributions, setShowAttributions] = useState({})  // Attribution visibility
const [attributionData, setAttributionData] = useState({})    // Attribution data
```

**Key Functions**:
- `useApiHealth()` - Checks API server status
- `fetchAttributions()` - Fetches contributor data
- `sendMessage()` - Sends query to API
- `clearChat()` - Clears chat history
- `renderMessageContent()` - Renders messages with attributions

---

### 2. UI Components (src/components/ui/)

All UI components are built with **Radix UI** primitives and styled with **Tailwind CSS**.

#### Button (button.tsx)
```jsx
<Button variant="default" size="md">Click Me</Button>
```

**Variants**: `default`, `destructive`, `outline`, `secondary`, `ghost`, `link`  
**Sizes**: `default`, `sm`, `lg`, `icon`

#### Card (card.tsx)
```jsx
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
  </CardHeader>
  <CardContent>Content</CardContent>
</Card>
```

#### Input (input.tsx)
```jsx
<Input type="text" placeholder="Enter text..." />
```

#### Badge (badge.tsx)
```jsx
<Badge variant="default">New</Badge>
```

**Variants**: `default`, `secondary`, `destructive`, `outline`

#### ScrollArea (scroll-area.tsx)
```jsx
<ScrollArea className="h-96">
  {/* Scrollable content */}
</ScrollArea>
```

#### Progress (progress.tsx)
```jsx
<Progress value={50} max={100} />
```

#### Separator (separator.tsx)
```jsx
<Separator orientation="horizontal" />
```

---

## Features

### 1. Chat Interface
**Description**: Real-time chat interface for asking OSRS questions

**Components Used**:
- `ScrollArea` - Scrollable message list
- `Card` - Message containers
- `Input` - Query input field
- `Button` - Send button
- `Badge` - Status indicators

**Features**:
- Auto-scroll to latest message
- Loading indicator during API calls
- Progress bar for long queries
- Message persistence in localStorage
- Clear chat button

**Message Types**:
- **User Messages**: Questions from the user
- **AI Messages**: Answers from the RAG system
- **System Messages**: Welcome message

---

### 2. Attribution System
**Description**: Shows wiki sources and contributors for AI answers with detailed revision information

**How It Works**:
1. AI includes citation markers in response: `[CITE:source="Page"|text="exact text"]paraphrased[/CITE]`
2. Frontend parses citations and highlights text
3. User clicks "Show Attributions" button
4. API fetches contributor data from MediaWiki REST API with revision caching
5. Highlighted text displays with interactive tooltips

**Recent Updates (October 2025)**:
- ✅ Removed "Agent Actions" section (no longer shows tool calls)
- ✅ Removed "Agent Reasoning" section (no longer shows reasoning steps)
- ✅ Added "📚 Sources Used" section showing wiki pages consulted
- ✅ Increased timeout from 2 minutes to 5 minutes for complex queries
- ✅ Improved citation format handling with explicit placement rules

**Enhanced Tooltip Contents** (Updated October 2025):
- 📖 **Source Page**: Clickable link to wiki page
- 📝 **Excerpt**: Exact text from wiki (truncated to 100 chars)
- ✍️ **Author**: Wiki contributor username (highlighted in yellow)
- 🏆 **Original Author Badge**: Green badge if they introduced the text
- 📅 **Timestamp**: Date and time of the revision (formatted)
- 💬 **Edit Comment**: Edit summary provided by contributor (truncated to 80 chars)
- 🔗 **Revision Link**: Direct link to the specific wiki revision

**Visual Structure**:
```
┌─────────────────────────────────────────────────────┐
│ 📖 Dragon Slayer I                                  │ ← Wiki page link
├─────────────────────────────────────────────────────┤
│ "You must have 32 Quest points to start this..."   │ ← Excerpt (truncated)
├─────────────────────────────────────────────────────┤
│ ✍️ Telemonke (Original Author)                     │ ← Author + badge
│ 📅 Sep 17, 2025, 07:16 PM                          │ ← Timestamp
│ 💬 "improved the description for which door..."    │ ← Edit comment
├─────────────────────────────────────────────────────┤
│ 🔗 View Revision →                                  │ ← Revision link
└─────────────────────────────────────────────────────┘
```

**Implementation**:
```javascript
// Highlight text with enhanced attribution tooltip
<span className="bg-yellow-400/20 border-b-2 border-yellow-400/50 cursor-help relative group">
  {text}
  <span className="opacity-0 group-hover:opacity-100 hover:opacity-100 absolute bottom-full left-0 mb-2 w-80 p-3 bg-gray-900 border border-yellow-400/50 rounded-lg shadow-xl text-xs z-50 transition-opacity duration-200 pointer-events-none group-hover:pointer-events-auto hover:pointer-events-auto">
    {/* Source page link */}
    <a href={attr.source_url} target="_blank" rel="noreferrer" className="font-semibold text-yellow-300 hover:text-yellow-200 mb-1 block pointer-events-auto">
      📖 {attr.source_title}
    </a>

    {/* Excerpt */}
    <div className="text-gray-300 mb-2 italic text-[11px]">
      "{attr.excerpt.substring(0, 100)}{attr.excerpt.length > 100 ? '...' : ''}"
    </div>

    {/* Attribution details */}
    <div className="border-t border-gray-700 pt-2 mb-2">
      {/* Author */}
      <div className="text-gray-400 text-xs mb-1">
        <span className="text-yellow-300">✍️ {attr.author}</span>
        {attr.is_original_author && <span className="ml-1 text-green-400">(Original Author)</span>}
      </div>

      {/* Timestamp */}
      {attr.timestamp && (
        <div className="text-gray-500 text-[10px] mb-1">
          📅 {new Date(attr.timestamp).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
          })}
        </div>
      )}

      {/* Edit comment */}
      {attr.comment && attr.comment !== '(No edit summary provided)' && (
        <div className="text-gray-500 text-[10px] mb-1 italic">
          💬 "{attr.comment.substring(0, 80)}{attr.comment.length > 80 ? '...' : ''}"
        </div>
      )}
    </div>

    {/* Revision link */}
    {attr.revision_url && (
      <a href={attr.revision_url} target="_blank" rel="noreferrer" className="text-blue-400 hover:text-blue-300 text-xs inline-block pointer-events-auto">
        🔗 View Revision →
      </a>
    )}
  </span>
</span>
```

**Attribution Data Structure**:
```javascript
{
  text: "Zulrah has a combat level of 725",
  start: 0,
  end: 33,
  source_title: "Zulrah",
  source_url: "https://oldschool.runescape.wiki/w/Zulrah",
  excerpt: "Combat Level: 725",
  author: "Microbrews",
  timestamp: "2024-01-15T10:30:00Z",
  revision_url: "https://oldschool.runescape.wiki/w/index.php?title=Zulrah&oldid=14997277",
  is_original_author: true,
  comment: "Updated combat stats",
  revision_id: 14997277
}
```

---

### 3. Economic Dashboard
**Description**: Track Grand Exchange prices and analyze market trends

**Components Used**:
- `Card` - Dashboard container
- `Tabs` - Switch between views (Price History, Multi-Item Comparison, Trends)
- `Input` - Item name search
- `Button` - Add/remove items, refresh data
- `Progress` - Loading indicator
- `Badge` - Price change indicators

**Features**:
- **Price History Charts**: View price trends over time (7d, 30d, 90d, 1y)
- **Multi-Item Comparison**: Compare prices of multiple items side-by-side
- **Trend Analysis**: Calculate price changes, volatility, and trends
- **Real-time Updates**: Fetch latest prices from API
- **Responsive Charts**: Interactive charts with hover tooltips

**Implementation** (`src/components/EconomicDashboard.jsx`):
```jsx
export default function EconomicDashboard() {
  const [items, setItems] = useState(['Abyssal whip'])
  const [priceData, setPriceData] = useState({})
  const [timeRange, setTimeRange] = useState('7d')
  const [loading, setLoading] = useState(false)

  // Fetch price history for an item
  async function fetchPriceHistory(itemName, range) {
    const response = await fetch(`${API_BASE}/price_history?item=${itemName}&range=${range}`)
    const data = await response.json()
    return data.history
  }

  // Calculate trend (up/down/stable)
  function calculateTrend(history) {
    if (history.length < 2) return 'stable'
    const recent = history.slice(-10)
    const avg = recent.reduce((sum, p) => sum + p.price, 0) / recent.length
    const latest = recent[recent.length - 1].price
    const change = ((latest - avg) / avg) * 100
    if (change > 5) return 'up'
    if (change < -5) return 'down'
    return 'stable'
  }

  // Render price chart
  function renderChart(history) {
    // Simple ASCII chart or use a charting library
    return (
      <div className="space-y-1">
        {history.map((point, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-xs text-gray-400">{point.timestamp}</span>
            <div
              className="h-2 bg-blue-500 rounded"
              style={{ width: `${(point.price / maxPrice) * 100}%` }}
            />
            <span className="text-sm">{formatPrice(point.price)}</span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>📊 Economic Dashboard</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="history">
          <TabsList>
            <TabsTrigger value="history">Price History</TabsTrigger>
            <TabsTrigger value="compare">Compare Items</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
          </TabsList>

          <TabsContent value="history">
            {/* Price history chart */}
          </TabsContent>

          <TabsContent value="compare">
            {/* Multi-item comparison */}
          </TabsContent>

          <TabsContent value="trends">
            {/* Trend analysis */}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
```

**API Endpoints Used**:
- `GET /price_history?item=<name>&range=<7d|30d|90d|1y>` - Get price history
- `GET /item_price?item=<name>` - Get current price

**Data Structure**:
```javascript
{
  item_name: "Abyssal whip",
  history: [
    { timestamp: "2025-10-01T10:00:00Z", price: 1498814 },
    { timestamp: "2025-10-02T10:00:00Z", price: 1502341 },
    // ...
  ],
  stats: {
    avg_price: 1500000,
    min_price: 1450000,
    max_price: 1550000,
    volatility: 0.034,
    trend: "up"
  }
}
```

**Visual Design**:
- Blue theme for price charts
- Green badges for price increases
- Red badges for price decreases
- Gray badges for stable prices
- Responsive grid layout for multi-item comparison

---

### 4. Sources Used Section
**Description**: Displays all wiki pages consulted by the AI during research

**Visual Design**:
- Purple background (`bg-purple-500/5`)
- Clickable wiki page links
- Displayed below AI response
- Always visible (no toggle required)

**Implementation**:
```javascript
{m.sources?.length ? (
  <div className="mt-3 border rounded-md p-3 bg-purple-500/5">
    <div className="text-xs text-purple-300 mb-2 font-medium">📚 Sources Used</div>
    <div className="flex flex-wrap gap-2">
      {m.sources.map((s, i) => (
        <a
          key={i}
          className="text-blue-300 hover:text-blue-200 underline text-sm"
          href={s.url}
          target="_blank"
          rel="noreferrer"
          title={s.title}
        >
          {s.title}
        </a>
      ))}
    </div>
  </div>
) : null}
```

**Data Structure**:
```javascript
sources: [
  {
    title: "Zulrah",
    url: "https://oldschool.runescape.wiki/w/Zulrah"
  },
  {
    title: "Toxic blowpipe",
    url: "https://oldschool.runescape.wiki/w/Toxic_blowpipe"
  }
]
```

**Purpose**:
- Shows transparency about AI's research process
- Provides quick access to source material
- Complements the attribution system (which shows specific contributors)

---

### 5. API Health Check
**Description**: Real-time status indicator for API server

**Implementation**:
```javascript
function useApiHealth() {
  const [status, setStatus] = useState('checking')
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setStatus(d?.status === 'healthy' ? 'online' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [])
  return status
}
```

**Status Indicators**:
- 🟢 **Online**: API server is healthy
- 🔴 **Offline**: API server is unreachable
- 🟡 **Checking**: Initial connection check

---

### 6. LocalStorage Persistence
**Description**: Chat history persists across page refreshes

**Implementation**:
```javascript
// Load messages on mount
const [messages, setMessages] = useState(() => {
  const saved = localStorage.getItem('osrs-chat-messages')
  return saved ? JSON.parse(saved) : [defaultMessage]
})

// Save messages on change
useEffect(() => {
  localStorage.setItem('osrs-chat-messages', JSON.stringify(messages))
}, [messages])
```

**Storage Key**: `osrs-chat-messages`

---

### 7. Progressive Web App (PWA)
**Description**: Install as standalone app on any device

**Features**:
- **Offline Support**: Service worker caches assets
- **Install Prompt**: Add to home screen
- **App-Like Experience**: Standalone display mode
- **Theme Colors**: Custom theme for status bar

**Manifest** (`public/manifest.json`):
```json
{
  "name": "OSRS AI Assistant",
  "short_name": "OSRS AI",
  "description": "Old School RuneScape AI Assistant with RAG",
  "theme_color": "#1e293b",
  "background_color": "#0f172a",
  "display": "standalone"
}
```

---

## Styling System

### Tailwind CSS Configuration
**File**: `tailwind.config.js`

**Custom Theme**:
- **Dark Mode**: Class-based dark mode support
- **Custom Colors**: OSRS-themed color palette
- **Border Radius**: Consistent border radius variables
- **Animations**: Custom animation utilities

**Color Palette**:
```javascript
colors: {
  primary: "hsl(45 90% 55%)",      // OSRS gold accent
  background: "hsl(240 10% 3.9%)", // Dark background
  foreground: "hsl(0 0% 98%)",     // Light text
  card: "hsl(240 10% 3.9%)",       // Card background
  border: "hsl(240 3.7% 15.9%)",   // Border color
}
```

---

### Global Styles
**File**: `src/index.css`

**Features**:
- Tailwind base, components, utilities
- Custom CSS variables for theming
- Gradient background
- Full-height layout

**Background Gradient**:
```css
body {
  background:
    radial-gradient(800px at 20% 0%, rgba(200,170,110,0.08), transparent 60%),
    linear-gradient(135deg, #0b1020 0%, #111827 100%);
}
```

---

### Component Styling
**Pattern**: Radix UI + Tailwind + CVA (Class Variance Authority)

**Example** (Button component):
```typescript
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 ...",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground ...",
        outline: "border border-input bg-background ...",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
      },
    },
  }
)
```

---

### Utility Function
**File**: `src/lib/utils.ts`

**cn() Function**: Merges Tailwind classes intelligently
```typescript
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

**Usage**:
```jsx
<div className={cn("base-class", condition && "conditional-class")} />
```

---

## State Management

### React Hooks Used

#### useState
```javascript
const [messages, setMessages] = useState([])
const [loading, setLoading] = useState(false)
```

#### useEffect
```javascript
// Auto-scroll on message change
useEffect(() => {
  scrollToBottom()
}, [messages])

// Save to localStorage
useEffect(() => {
  localStorage.setItem('key', JSON.stringify(data))
}, [data])
```

#### useRef
```javascript
const listRef = useRef(null)
// Access DOM element: listRef.current
```

#### Custom Hooks
```javascript
// API health check
const status = useApiHealth()

// Toast notifications (from shadcn/ui)
const { toast } = useToast()
```

---

### State Structure

**Messages Array**:
```javascript
[
  {
    role: 'ai',
    id: 1234567890,
    content: 'Hello! I\'m your OSRS AI assistant.',
    sources: [],
    citations: []
  },
  {
    role: 'user',
    id: 1234567891,
    content: 'What is Zulrah?'
  },
  {
    role: 'ai',
    id: 1234567892,
    content: 'Zulrah is a level 725 snake boss...',
    sources: [{title: 'Zulrah', url: '...'}],
    citations: [{text: '...', source_title: 'Zulrah', ...}]
  }
]
```

**Note**: `reasoning` and `tool_calls` fields were removed in October 2025 update to simplify the UI.

**Attribution Data**:
```javascript
{
  [messageId]: [
    {
      text: 'Zulrah has a combat level of 725',
      start: 0,
      end: 33,
      source_title: 'Zulrah',
      source_url: 'https://oldschool.runescape.wiki/w/Zulrah',
      excerpt: 'Combat Level: 725',
      author: 'Microbrews',
      timestamp: '2024-01-15T10:30:00Z',
      revision_url: 'https://...',
      is_original_author: true
    }
  ]
}
```

---

## API Integration

### API Base URL
```javascript
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001'
```

### Endpoints Used

#### 1. Health Check
```javascript
GET /health

Response:
{
  "status": "healthy",
  "service": "OSRS RAG API",
  "version": "3.0"
}
```

#### 2. Chat Query
```javascript
POST /chat
Content-Type: application/json

Body:
{
  "query": "What is Zulrah?"
}

Response:
{
  "success": true,
  "response": "Zulrah is a level 725 snake boss...",
  "sources": [...],
  "citations": [...]
}

Note: `reasoning` and `tool_calls` fields removed in October 2025 update.
```

#### 3. Attributions
```javascript
POST /attributions
Content-Type: application/json

Body:
{
  "citations": [...]
}

Response:
{
  "success": true,
  "attributions": [...]
}
```

---

### Error Handling
```javascript
try {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal: AbortSignal.timeout(300000)  // 5 minute timeout (increased from 2 minutes)
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const data = await response.json()

  if (!data.success) {
    throw new Error(data.error || 'Unknown error')
  }

  // Handle success
} catch (error) {
  if (error.name === 'TimeoutError') {
    console.error('Request timed out after 5 minutes')
  } else {
    console.error('API Error:', error)
  }
  // Show error message to user
}
```

**Timeout Configuration**:
- **Chat Queries**: 5 minutes (300 seconds) - increased in October 2025 for complex queries
- **Attribution Lookups**: 30 seconds
- **Health Checks**: 5 seconds

---

## PWA Features

### Service Worker
**File**: Automatically generated by `vite-plugin-pwa`

**Caching Strategy**:
- **Precache**: All static assets (JS, CSS, HTML, images)
- **Runtime Cache**: Google Fonts
- **Cache-First**: Fonts and static resources
- **Network-First**: API requests (not cached)

**Configuration** (`vite.config.js`):
```javascript
VitePWA({
  registerType: 'autoUpdate',
  includeAssets: ['vite.svg'],
  workbox: {
    globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2}'],
    runtimeCaching: [
      {
        urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
        handler: 'CacheFirst',
        options: {
          cacheName: 'google-fonts-cache',
          expiration: { maxAgeSeconds: 60 * 60 * 24 * 365 }
        }
      }
    ]
  }
})
```

---

### Installation
**Desktop**: Chrome/Edge will show install prompt in address bar
**Mobile**: "Add to Home Screen" option in browser menu

**Install Criteria**:
- ✅ Served over HTTPS (or localhost)
- ✅ Has valid manifest.json
- ✅ Has service worker registered
- ✅ Has icons defined

---

### Offline Support
**What Works Offline**:
- UI loads and displays
- Previously cached messages (localStorage)
- Static assets (CSS, JS, images)

**What Doesn't Work Offline**:
- New API queries (requires internet)
- Attribution lookups (requires MediaWiki API)
- Health check (requires API server)

---

## Configuration

### Vite Configuration
**File**: `vite.config.js`

**Key Settings**:
```javascript
export default defineConfig({
  plugins: [react(), VitePWA(...)],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),  // Import alias
    },
  },
})
```

**Import Alias**: Use `@/` to import from `src/`
```javascript
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
```

---

### Tailwind Configuration
**File**: `tailwind.config.js`

**Content Paths**:
```javascript
content: [
  "./index.html",
  "./src/**/*.{js,jsx,ts,tsx}"
]
```

**Dark Mode**: Class-based
```javascript
darkMode: ["class"]
```

---

### ESLint Configuration
**File**: `eslint.config.js`

**Rules**: React best practices, hooks rules, refresh rules

---

### TypeScript Configuration
**File**: `tsconfig.json`

**Purpose**: Type checking for `.tsx` files (UI components)

**Note**: Main app is `.jsx` (JavaScript), UI components are `.tsx` (TypeScript)

---

## Development

### Adding New UI Components
1. Create component in `src/components/ui/`
2. Use Radix UI primitives
3. Style with Tailwind CSS
4. Export from component file

**Example**:
```typescript
// src/components/ui/my-component.tsx
import * as React from "react"
import { cn } from "@/lib/utils"

export interface MyComponentProps {
  className?: string
  children?: React.ReactNode
}

const MyComponent = React.forwardRef<HTMLDivElement, MyComponentProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn("base-styles", className)}
        {...props}
      >
        {children}
      </div>
    )
  }
)
MyComponent.displayName = "MyComponent"

export { MyComponent }
```

---

### Adding New Features to App.jsx
1. Add state variables with `useState`
2. Add effect hooks with `useEffect`
3. Add handler functions
4. Update JSX to use new features

**Example** (Adding a new button):
```javascript
// Add state
const [myFeature, setMyFeature] = useState(false)

// Add handler
function handleMyFeature() {
  setMyFeature(!myFeature)
}

// Add to JSX
<Button onClick={handleMyFeature}>
  {myFeature ? 'Disable' : 'Enable'} Feature
</Button>
```

---

### Debugging

#### React DevTools
Install React DevTools browser extension to inspect:
- Component tree
- Props and state
- Performance profiling

#### Console Logging
```javascript
console.log('Debug:', variable)
console.error('Error:', error)
console.table(arrayOfObjects)
```

#### Network Tab
Monitor API requests in browser DevTools:
- Check request/response
- Verify status codes
- Inspect headers and body

---

### Hot Module Replacement (HMR)
Vite provides instant updates without full page reload:
- Edit `.jsx` files → Components update instantly
- Edit `.css` files → Styles update instantly
- State is preserved during updates

---

## Build & Deployment

### Production Build
```bash
npm run build
```

**Output**: `dist/` directory

**Contents**:
- `index.html` - Entry HTML
- `assets/` - Bundled JS, CSS, images
- `manifest.json` - PWA manifest
- `sw.js` - Service worker

**Optimizations**:
- Minified JavaScript
- Minified CSS
- Tree-shaking (removes unused code)
- Code splitting
- Asset optimization

---

### Preview Production Build
```bash
npm run preview
```

**Access**: http://localhost:4173

---

### Deployment Options

#### 1. Static Hosting (Recommended)
Deploy `dist/` folder to:
- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop `dist/` folder
- **GitHub Pages**: Push to `gh-pages` branch
- **Cloudflare Pages**: Connect GitHub repo

#### 2. Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "5173"]
EXPOSE 5173
```

Build and run:
```bash
docker build -t osrs-frontend .
docker run -p 5173:5173 osrs-frontend
```

#### 3. Nginx
```nginx
server {
  listen 80;
  server_name osrs-ai.local;
  root /var/www/osrs-frontend/dist;
  index index.html;

  location / {
    try_files $uri $uri/ /index.html;
  }
}
```

---

### Environment Variables for Production
Create `.env.production`:
```bash
VITE_API_BASE=https://api.osrs-ai.com
```

Build with production env:
```bash
npm run build
```

---

## Performance Optimization

### Code Splitting
Vite automatically splits code by route/component:
- Main bundle: Core app code
- Vendor bundle: node_modules
- Component bundles: Lazy-loaded components

### Asset Optimization
- **Images**: Optimized and compressed
- **Fonts**: Subset and cached
- **Icons**: SVG sprites

### Lazy Loading
```javascript
import { lazy, Suspense } from 'react'

const HeavyComponent = lazy(() => import('./HeavyComponent'))

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  )
}
```

### Memoization
```javascript
import { useMemo, useCallback } from 'react'

// Memoize expensive calculations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data)
}, [data])

// Memoize callback functions
const handleClick = useCallback(() => {
  doSomething(value)
}, [value])
```

---

## Testing

### Manual Testing Checklist
- [ ] Chat interface loads
- [ ] Can send messages
- [ ] Messages persist after refresh
- [ ] API health indicator works
- [ ] Attributions button appears
- [ ] Tooltips show on hover
- [ ] Links in tooltips are clickable
- [ ] Clear chat button works
- [ ] Responsive on mobile
- [ ] PWA install prompt appears

### Browser Testing
Test in:
- Chrome/Edge (Chromium)
- Firefox
- Safari (macOS/iOS)
- Mobile browsers

---

## Troubleshooting

### Issue: "Cannot connect to API"
**Solution**: Ensure API server is running on port 5001
```bash
cd /Users/brandon/Documents/projects/GE/api
python3 osrs_api_server.py --host 0.0.0.0
```

### Issue: "Module not found"
**Solution**: Reinstall dependencies
```bash
rm -rf node_modules package-lock.json
npm install
```

### Issue: "Port 5173 already in use"
**Solution**: Kill process or use different port
```bash
lsof -i :5173
kill -9 <PID>

# Or use different port
npm run dev -- --port 3000
```

### Issue: "Styles not loading"
**Solution**: Rebuild Tailwind
```bash
npm run build
```

### Issue: "PWA not installing"
**Solution**: Check requirements
- Must be HTTPS or localhost
- Check manifest.json is valid
- Check service worker is registered
- Check browser console for errors

---

## Browser Support

### Minimum Versions
- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### Mobile Support
- **iOS Safari**: 14+
- **Chrome Mobile**: 90+
- **Samsung Internet**: 14+

### Features Requiring Modern Browsers
- CSS Grid
- CSS Custom Properties
- ES6+ JavaScript
- Fetch API
- LocalStorage
- Service Workers

---

## Accessibility

### ARIA Support
Radix UI components include built-in ARIA attributes:
- `role` attributes
- `aria-label` attributes
- `aria-expanded` for collapsible elements
- `aria-hidden` for decorative elements

### Keyboard Navigation
- **Tab**: Navigate between interactive elements
- **Enter/Space**: Activate buttons
- **Escape**: Close dialogs/tooltips
- **Arrow Keys**: Navigate tabs

### Screen Reader Support
All interactive elements have proper labels and descriptions.

---

## Credits

**Developed by**: Brandon Inkel
**UI Components**: Radix UI + shadcn/ui
**Styling**: Tailwind CSS
**Build Tool**: Vite
**Framework**: React

---

## Support

For issues or questions:
1. Check browser console for errors
2. Verify API server is running
3. Check network tab for failed requests
4. Review this documentation
5. Clear browser cache and localStorage

---

**Last Updated**: October 5, 2025
**Version**: 1.1.0
**License**: Private/Internal Use


