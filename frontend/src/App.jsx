import { useEffect, useRef, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5003'

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
  const [messages, setMessages] = useState([
    { role: 'ai', content: `Hello! I'm your OSRS AI assistant. Ask me anything about Old School RuneScape!` }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [genInfo, setGenInfo] = useState(null) // {context_tokens, response_tokens}
  const listRef = useRef(null)

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages, loading])

  async function sendMessage() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)
    setProgress(0)
    setGenInfo(null)
    setMessages(m => [...m, { role: 'user', content: q }])

    const loadingId = Date.now()
    setMessages(m => [...m, { role: 'loading', id: loadingId, content: 'Thinking…' }])

    try {
      const resp = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 20, show_sources: true })
      })
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body.getReader()
      const dec = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = dec.decode(value)
        for (const raw of chunk.split('\n')) {
          if (!raw.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(raw.slice(6))
            if (typeof evt.progress === 'number') {
              setProgress(Math.max(0, Math.min(100, evt.progress)))
            }
            if (evt.stage === 'metrics' && evt.metrics) {
              setGenInfo(evt.metrics)
            }
            if (evt.stage === 'complete' && evt.result) {
              const { response, sources, similarity_scores, excerpts } = evt.result
              setMessages(m => m.filter(x => x.id !== loadingId))
              setMessages(m => [...m, { role: 'ai', content: response, sources, scores: similarity_scores, excerpts }])
            } else if (evt.stage === 'error') {
              setMessages(m => m.filter(x => x.id !== loadingId))
              setMessages(m => [...m, { role: 'ai', content: `❌ Error: ${evt.message}` }])
            }
          } catch {}
        }
      }
    } catch (e) {
      setMessages(m => m.filter(x => x.id !== loadingId))
      setMessages(m => [...m, { role: 'ai', content: `❌ Error: ${e.message}` }])
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-[#0b1020] text-slate-100">
      <div className="max-w-6xl mx-auto px-4 pt-6 space-y-6">
        <Card className="bg-background/60 border-amber-400/20">
          <CardHeader className="text-center">
            <CardTitle className="text-amber-300 text-2xl md:text-3xl tracking-wide">
              OSRS AI Assistant
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div className="flex flex-col gap-1 rounded-md p-3 border bg-card/50">
                <div className="text-xs text-muted-foreground">RAG API</div>
                <div className="flex items-center gap-2">
                  <span className={clsx('inline-block w-2.5 h-2.5 rounded-full', status==='online'?'bg-emerald-500':'bg-red-600')} />
                  <span>{status==='online'? 'Connected' : status==='checking' ? 'Checking…' : 'Disconnected'}</span>
                </div>
              </div>
              <div className="flex flex-col gap-1 rounded-md p-3 border bg-card/50">
                <div className="text-xs text-muted-foreground">AI Model</div>
                <div><Badge variant="secondary">LLaMA 3.1 8B</Badge> + RAG</div>
              </div>
              <div className="flex flex-col gap-1 rounded-md p-3 border bg-card/50">
                <div className="text-xs text-muted-foreground">Embeddings</div>
                <div>Wiki</div>
              </div>
              <div className="flex flex-col gap-1 rounded-md p-3 border bg-card/50">
                <div className="text-xs text-muted-foreground">Theme</div>
                <div>OSRS Dark</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-background/60 border-amber-400/20 overflow-hidden">
          <CardHeader className="py-3">
            <CardTitle className="text-[#c8aa6e] text-base">Chat</CardTitle>
          </CardHeader>
          <Separator />
          <CardContent className="p-0">
            <div className="grid grid-rows-[1fr_auto] h-[65vh]">
              <ScrollArea className="h-full" ref={listRef}>
                <div className="p-5 space-y-4">
                  {messages.map((m, idx) => (
                    <div key={m.id || idx} className={clsx('rounded-xl p-4 border shadow', m.role==='user'? 'bg-blue-500/10 border-blue-500/40' : m.role==='loading'? 'bg-amber-400/10 border-amber-400/50' : 'bg-emerald-500/10 border-emerald-500/40')}>
                      <div className="text-xs opacity-70 mb-1">{m.role==='user'?'You':'OSRS AI'}</div>
                      <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                      {m.sources?.length ? (
                        <div className="mt-3 border rounded-md p-3">
                          <div className="text-sm text-purple-300 mb-2">Sources</div>
                          <div className="space-y-2">
                            {m.sources.map((s, i) => (
                              <div key={i} className="flex justify-between items-center bg-white/5 border rounded px-2 py-1 text-sm">
                                <span className="truncate mr-2">{s.title}</span>
                                <div className="flex items-center gap-3">
                                  {m.scores?.[i] != null && (
                                    <Badge variant="secondary">{(m.scores[i]*100).toFixed(1)}%</Badge>
                                  )}
                                  {s.url && <a className="text-blue-300 hover:text-blue-200 underline" href={s.url} target="_blank" rel="noreferrer">Open</a>}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              </ScrollArea>
              <div className="p-4 border-t bg-background/80">
                <div className="flex gap-2">
                  <Input
                    placeholder="Ask me anything about OSRS…"
                    value={input}
                    onChange={e=>setInput(e.target.value)}
                    onKeyDown={e=>{ if(e.key==='Enter' && !e.shiftKey) sendMessage() }}
                    disabled={loading}
                  />
                  <Button onClick={sendMessage} disabled={loading} className="bg-primary text-primary-foreground">
                    {loading? 'Thinking…' : 'Send'}
                  </Button>
                </div>
                {loading ? (
                  <div className="mt-3 flex items-center gap-3">
                    <Progress value={progress} className="h-2 w-full" />
                    {genInfo?.response_tokens ? (
                      <span className="text-xs text-muted-foreground whitespace-nowrap">{progress}%</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
