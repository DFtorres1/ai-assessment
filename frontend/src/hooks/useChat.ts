import { useCallback, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import type { Citation, Message, ToolCall, UserType } from '../types'

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

const uid = () => Math.random().toString(36).slice(2, 10)

const getOrCreateSessionId = () => {
  const key = 'blossom_session_id'
  let id = sessionStorage.getItem(key)
  if (!id) { id = `web-${uid()}`; sessionStorage.setItem(key, id) }
  return id
}

const useChat = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const sessionId = useRef(getOrCreateSessionId())

  const send = useCallback((text: string, userType: UserType) => {
    if (!text.trim() || isLoading) return

    setIsLoading(true)

    const userMsg: Message = { id: uid(), role: 'user', text: text.trim(), citations: [], tool_calls: [], streaming: false }
    const botId = uid()
    const botMsg: Message = { id: botId, role: 'assistant', text: '', citations: [], tool_calls: [], streaming: true }

    setMessages(prev => [...prev, userMsg, botMsg])

    const params = new URLSearchParams({ session_id: sessionId.current, message: text.trim(), user_type: userType })
    const es = new EventSource(`${API_BASE}/chat/stream?${params.toString()}`)

    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const data = JSON.parse(e.data) as {
          type: string; content?: string; citations?: Citation[]; tool_calls?: ToolCall[]; message?: string
        }
        if (data.type === 'citations') {
          setMessages(prev => prev.map(m => m.id === botId ? { ...m, citations: data.citations ?? [] } : m))
        } else if (data.type === 'token') {
          flushSync(() => {
            setMessages(prev => prev.map(m => m.id === botId ? { ...m, text: m.text + (data.content ?? '') } : m))
          })
        } else if (data.type === 'done') {
          setMessages(prev => prev.map(m => m.id === botId ? { ...m, streaming: false, tool_calls: data.tool_calls ?? [] } : m))
          es.close()
          setIsLoading(false)
        } else if (data.type === 'error') {
          setMessages(prev => prev.map(m => m.id === botId ? { ...m, streaming: false, error: data.message ?? 'Something went wrong.' } : m))
          es.close()
          setIsLoading(false)
        }
      } catch { /* malformed SSE frame */ }
    }

    es.onerror = () => {
      setMessages(prev => prev.map(m => m.id === botId
        ? { ...m, streaming: false, error: 'Connection lost. Is the backend running?' } : m))
      es.close()
      setIsLoading(false)
    }
  }, [isLoading])

  return { messages, isLoading, send }
}

export default useChat
