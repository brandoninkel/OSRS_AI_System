import { useEffect, useRef, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import EconomicDashboard from '@/components/EconomicDashboard'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001'

function useApiHealth() {
  const [status, setStatus] = useState('checking')
  useEffect(() => {
    let mounted = true
    fetch(`${API_BASE}/health`).then(r => r.json()).then(d => {
      if (!mounted) return
      setStatus(d?.status === 'healthy' ? 'online' : 'offline')
    }).catch(() => setStatus('offline'))
    return () => { mounted = false }
  }, [])
  return status
}

function clsx(...a) { return a.filter(Boolean).join(' ') }

export default function App() {
  const status = useApiHealth()

  // Load messages from localStorage on mount
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('osrs-chat-messages')
      if (saved) {
        return JSON.parse(saved)
      }
    } catch (e) {
      console.error('Failed to load messages from localStorage:', e)
    }
    return [{ role: 'ai', content: `Hello! I'm your OSRS AI assistant. Ask me anything about Old School RuneScape!` }]
  })

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [genInfo, setGenInfo] = useState(null) // {context_tokens, response_tokens}
  const [showAttributions, setShowAttributions] = useState({}) // Track attribution visibility per message
  const [attributionData, setAttributionData] = useState({}) // Store attribution data per message
  const [loadingAttributions, setLoadingAttributions] = useState({}) // Track loading state
  const listRef = useRef(null)

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('osrs-chat-messages', JSON.stringify(messages))
    } catch (e) {
      console.error('Failed to save messages to localStorage:', e)
    }
  }, [messages])

  // Auto-scroll to bottom when messages change or loading state changes
  useEffect(() => {
    if (listRef.current) {
      const scrollElement = listRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight
      }
    }
  }, [messages, loading, progress])

  function clearChat() {
    setMessages([{ role: 'ai', content: `Hello! I'm your OSRS AI assistant. Ask me anything about Old School RuneScape!` }])
    localStorage.removeItem('osrs-chat-messages')
  }

  async function fetchAttributions(msgId, citations) {
    if (loadingAttributions[msgId] || attributionData[msgId]) {
      return // Already loading or loaded
    }

    setLoadingAttributions(prev => ({...prev, [msgId]: true}))

    try {
      const resp = await fetch(`${API_BASE}/attributions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ citations })
      })

      const data = await resp.json()

      if (data.success) {
        setAttributionData(prev => ({...prev, [msgId]: data.attributions}))
      }
    } catch (e) {
      console.error('Error fetching attributions:', e)
    } finally {
      setLoadingAttributions(prev => ({...prev, [msgId]: false}))
    }
  }

  async function sendMessage() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)
    setProgress(50) // Show progress while waiting
    setGenInfo(null)
    setMessages(m => [...m, { role: 'user', content: q }])

    const loadingId = Date.now()
    setMessages(m => [...m, { role: 'loading', id: loadingId, content: '🤖 Agent is thinking and searching...' }])

    // Set a timeout to prevent infinite "Thinking..." state
    const timeout = setTimeout(() => {
      setMessages(m => m.filter(x => x.id !== loadingId))
      setMessages(m => [...m, { role: 'ai', content: `❌ Request timed out. The agent may be processing a complex query. Please try again.` }])
      setLoading(false)
      setProgress(0)
    }, 300000) // 5 minute timeout for complex queries

    try {
      // Use non-streaming endpoint for V3 Agentic RAG
      const resp = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const data = await resp.json()

      clearTimeout(timeout)
      setMessages(m => m.filter(x => x.id !== loadingId))

      if (data.success) {
        // Add AI response with sources, reasoning, tool calls, and citations
        const msgId = Date.now()
        setMessages(m => [...m, {
          role: 'ai',
          id: msgId,
          content: data.response,
          sources: data.sources || [],
          reasoning: data.reasoning || [],
          tool_calls: data.tool_calls || [],
          citations: data.citations || []
        }])
      } else {
        setMessages(m => [...m, { role: 'ai', content: `❌ Error: ${data.error || 'Unknown error'}` }])
      }
    } catch (e) {
      clearTimeout(timeout)
      setMessages(m => m.filter(x => x.id !== loadingId))
      setMessages(m => [...m, { role: 'ai', content: `❌ Error: ${e.message}` }])
    }

    setLoading(false)
    setProgress(0)
  }

  return (
    <div className="flex flex-col h-screen bg-[#0b1020] text-slate-100">
      {/* Compact Header */}
      <div className="flex-none border-b border-amber-400/20 bg-background/60 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-amber-300 text-xl md:text-2xl font-semibold tracking-wide">
            OSRS AI Assistant
          </h1>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground hidden sm:inline">API</span>
            <span className={clsx('inline-block w-2.5 h-2.5 rounded-full', status==='online'?'bg-emerald-500':'bg-red-600')} />
            <span className="text-xs text-muted-foreground sm:hidden">
              {status==='online'? 'Online' : status==='checking' ? 'Checking' : 'Offline'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content with Tabs */}
      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="chat" className="h-full flex flex-col">
          <div className="flex-none border-b border-border bg-background/40">
            <div className="max-w-6xl mx-auto px-4">
              <TabsList className="bg-transparent border-0 h-12">
                <TabsTrigger value="chat" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                  💬 Chat
                </TabsTrigger>
                <TabsTrigger value="economic" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                  📊 Economic Dashboard
                </TabsTrigger>
              </TabsList>
            </div>
          </div>

          <TabsContent value="chat" className="flex-1 m-0 overflow-hidden">
            {/* Chat Interface */}

      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-6xl mx-auto w-full">
        <Card className="flex-1 flex flex-col bg-background/60 border-amber-400/20 m-4 overflow-hidden">
          <CardHeader className="flex-none py-3 px-4">
            <CardTitle className="text-[#c8aa6e] text-base">Chat</CardTitle>
          </CardHeader>
          <Separator />
          <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
            <ScrollArea className="flex-1" ref={listRef}>
              <div className="p-4 space-y-3 min-h-full">
                  {messages.map((m, idx) => {
                    const msgId = m.id || idx
                    const attributionsOn = showAttributions[msgId] || false

                    // Render answer with highlighted attributions
                    const renderAnswerWithAttributions = () => {
                      if (!attributionsOn || !attributionData[msgId] || attributionData[msgId].length === 0) {
                        return <div className="whitespace-pre-wrap leading-relaxed text-sm">{m.content}</div>
                      }

                      const attributions = attributionData[msgId]
                      const content = m.content

                      // Sort attributions by start position
                      const sorted = [...attributions].sort((a, b) => a.start - b.start)

                      const parts = []
                      let lastEnd = 0

                      sorted.forEach((attr, i) => {
                        // Add text before this attribution
                        if (attr.start > lastEnd) {
                          parts.push({
                            type: 'text',
                            content: content.substring(lastEnd, attr.start)
                          })
                        }

                        // Add attributed text with tooltip
                        parts.push({
                          type: 'attributed',
                          content: attr.text,
                          attribution: attr
                        })

                        lastEnd = attr.end
                      })

                      // Add remaining text
                      if (lastEnd < content.length) {
                        parts.push({
                          type: 'text',
                          content: content.substring(lastEnd)
                        })
                      }

                      // Define color palette for different attributions
                      const colors = [
                        { bg: 'bg-yellow-400/20', border: 'border-yellow-400/50', tooltip: 'border-yellow-400/50' },
                        { bg: 'bg-blue-400/20', border: 'border-blue-400/50', tooltip: 'border-blue-400/50' },
                        { bg: 'bg-green-400/20', border: 'border-green-400/50', tooltip: 'border-green-400/50' },
                        { bg: 'bg-purple-400/20', border: 'border-purple-400/50', tooltip: 'border-purple-400/50' },
                        { bg: 'bg-pink-400/20', border: 'border-pink-400/50', tooltip: 'border-pink-400/50' },
                        { bg: 'bg-orange-400/20', border: 'border-orange-400/50', tooltip: 'border-orange-400/50' },
                      ]

                      return (
                        <div className="whitespace-pre-wrap leading-relaxed text-sm">
                          {parts.map((part, i) => {
                            if (part.type === 'text') {
                              return <span key={i}>{part.content}</span>
                            } else {
                              const attr = part.attribution
                              const attrIndex = sorted.findIndex(a => a === attr)
                              const colorScheme = colors[attrIndex % colors.length]

                              return (
                                <span
                                  key={i}
                                  className={`${colorScheme.bg} border-b-2 ${colorScheme.border} cursor-help relative group`}
                                  style={{ display: 'inline-block' }}
                                >
                                  {part.content}
                                  <span
                                    className={`opacity-0 group-hover:opacity-100 hover:opacity-100 absolute bottom-full left-0 mb-2 w-80 p-3 bg-gray-900 border ${colorScheme.tooltip} rounded-lg shadow-xl text-xs z-50 transition-opacity duration-200 pointer-events-none group-hover:pointer-events-auto hover:pointer-events-auto`}
                                    style={{ transitionDelay: '0ms' }}
                                  >
                                    <a
                                      href={attr.source_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="font-semibold text-yellow-300 hover:text-yellow-200 mb-1 block pointer-events-auto"
                                    >
                                      📖 {attr.source_title}
                                    </a>
                                    <div className="text-gray-300 mb-2 italic text-[11px]">"{attr.excerpt.substring(0, 100)}{attr.excerpt.length > 100 ? '...' : ''}"</div>

                                    <div className="border-t border-gray-700 pt-2 mb-2">
                                      <div className="text-gray-400 text-xs mb-1">
                                        <span className="text-yellow-300">✍️ {attr.author}</span>
                                        {attr.is_original_author && <span className="ml-1 text-green-400">(Original Author)</span>}
                                      </div>

                                      {attr.timestamp && (
                                        <div className="text-gray-500 text-[10px] mb-1">
                                          📅 {new Date(attr.timestamp).toLocaleDateString('en-US', {
                                            year: 'numeric',
                                            month: 'short',
                                            day: 'numeric',
                                            hour: '2-digit',
                                            minute: '2-digit'
                                          })}
                                        </div>
                                      )}

                                      {attr.comment && attr.comment !== '(No edit summary provided)' && (
                                        <div className="text-gray-500 text-[10px] mb-1 italic">
                                          💬 "{attr.comment.substring(0, 80)}{attr.comment.length > 80 ? '...' : ''}"
                                        </div>
                                      )}
                                    </div>

                                    {attr.revision_url && (
                                      <a
                                        href={attr.revision_url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="text-blue-400 hover:text-blue-300 text-xs inline-block pointer-events-auto"
                                      >
                                        🔗 View Revision →
                                      </a>
                                    )}
                                  </span>
                                </span>
                              )
                            }
                          })}
                        </div>
                      )
                    }

                    return (
                    <div key={msgId} className={clsx('rounded-lg p-3 border shadow-sm', m.role==='user'? 'bg-blue-500/10 border-blue-500/40' : m.role==='loading'? 'bg-amber-400/10 border-amber-400/50' : 'bg-emerald-500/10 border-emerald-500/40')}>
                      <div className="text-xs opacity-70 mb-1.5 font-medium">{m.role==='user'?'You':'OSRS AI'}</div>
                      {renderAnswerWithAttributions()}

                      {/* Attribution toggle button - only show for AI responses with citations */}
                      {m.role === 'ai' && m.citations?.length > 0 && (
                        <div className="mt-3">
                          <Button
                            onClick={() => {
                              const newState = !attributionsOn
                              setShowAttributions(prev => ({...prev, [msgId]: newState}))
                              // Fetch attributions when turning on
                              if (newState && !attributionData[msgId]) {
                                fetchAttributions(msgId, m.citations)
                              }
                            }}
                            variant="outline"
                            size="sm"
                            className="text-xs h-7"
                            disabled={loadingAttributions[msgId]}
                          >
                            {loadingAttributions[msgId] ? '⏳ Loading...' : attributionsOn ? '🔍 Hide Attributions' : '🔍 Show Attributions'}
                          </Button>
                        </div>
                      )}

                      {/* Show sources used by AI */}
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


                    </div>
                  )})}
                </div>
              </ScrollArea>
            </CardContent>

            {/* Input Area - Fixed at bottom */}
            <div className="flex-none border-t bg-background/95 backdrop-blur">
              <div className="p-4 space-y-3">
                <div className="flex gap-2 items-center">
                  <Button
                    onClick={clearChat}
                    disabled={loading}
                    variant="outline"
                    size="sm"
                    className="text-xs h-7"
                  >
                    Clear
                  </Button>
                  <div className="text-xs text-muted-foreground">
                    {messages.length - 1} messages
                  </div>
                </div>
                <div className="flex gap-2">
                  <Input
                    placeholder="Ask me anything about OSRS…"
                    value={input}
                    onChange={e=>setInput(e.target.value)}
                    onKeyDown={e=>{ if(e.key==='Enter' && !e.shiftKey) sendMessage() }}
                    disabled={loading}
                    className="flex-1"
                  />
                  <Button onClick={sendMessage} disabled={loading} className="bg-primary text-primary-foreground px-6">
                    {loading? 'Thinking…' : 'Send'}
                  </Button>
                </div>
                {loading ? (
                  <div className="flex items-center gap-3">
                    <Progress value={progress} className="h-2 flex-1" />
                    {genInfo?.response_tokens ? (
                      <span className="text-xs text-muted-foreground whitespace-nowrap min-w-[3rem] text-right">{progress}%</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
        </Card>
      </div>
          </TabsContent>

          <TabsContent value="economic" className="flex-1 m-0 overflow-auto">
            <EconomicDashboard />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
