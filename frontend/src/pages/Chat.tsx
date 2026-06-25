import { useCallback, useEffect, useRef, useState } from 'react'
import { useHealthContext } from '../context/HealthContext'
import useChat from '../hooks/useChat'
import ChatBubble from '../components/ChatBubble'
import HealthBanner from '../components/HealthBanner'
import SuggestionChips from '../components/SuggestionChips'
import type { UserType } from '../types'

const Chat = () => {
  const { status } = useHealthContext()
  const { messages, isLoading, send } = useChat()
  const [input, setInput] = useState('')
  const [userType, setUserType] = useState<UserType>('member')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const backendDown = status === 'offline' || status === 'degraded' || status === 'no_api_key'
  const canSend = !!input.trim() && !isLoading && !backendDown

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(() => {
    if (!canSend) return
    const text = input
    setInput('')
    send(text, userType)
    setTimeout(() => inputRef.current?.focus(), 0)
  }, [canSend, input, userType, send])

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const onInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
  }

  const handleSuggestion = (text: string) => {
    if (backendDown || isLoading) return
    send(text, userType)
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <HealthBanner />

      <main className="flex-1 overflow-y-auto" aria-live="polite" aria-label="Chat messages">
        <div className="max-w-3xl mx-auto px-5 py-6 flex flex-col gap-5">
          {messages.length === 0
            ? <SuggestionChips onSelect={handleSuggestion} />
            : messages.map(msg => <ChatBubble key={msg.id} msg={msg} />)
          }
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 shadow-[0_-2px_8px_rgba(0,0,0,.05)] flex-shrink-0 px-5 pt-3.5 pb-2.5">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-2 mb-2.5">
            <span className="text-xs text-gray-400 font-medium">Role:</span>
            <div className="flex bg-gray-100 rounded-full p-0.5 gap-0.5">
              {(['member', 'staff'] as UserType[]).map(t => (
                <button
                  key={t}
                  onClick={() => setUserType(t)}
                  aria-pressed={userType === t}
                  className={`px-4 py-1 rounded-full text-[13px] font-medium transition-all border-none ${
                    userType === t
                      ? 'bg-white text-primary-700 font-bold shadow-sm'
                      : 'bg-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {t === 'member' ? 'Member' : 'Staff'}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2.5 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={onInput}
              onKeyDown={onKey}
              placeholder={
                status === 'no_api_key'
                  ? 'Set ANTHROPIC_API_KEY to enable the assistant…'
                  : backendDown
                    ? 'Backend is unavailable — please wait…'
                    : 'Ask about passwords, lockouts, MFA, or device recognition…'
              }
              rows={1}
              disabled={isLoading || backendDown}
              aria-label="Message input"
              className="flex-1 resize-none border-[1.5px] border-gray-200 rounded-2xl px-3.5 py-2.5 text-[15px] leading-relaxed bg-gray-50 text-gray-900 transition-colors max-h-[140px] overflow-y-auto disabled:opacity-50 disabled:cursor-not-allowed focus:border-primary-400"
            />
            <button
              onClick={handleSend}
              disabled={!canSend}
              aria-label="Send"
              className={`w-11 h-11 rounded-xl border-none flex items-center justify-center flex-shrink-0 transition-all ${
                canSend
                  ? 'bg-primary-700 text-white shadow-[0_2px_8px_rgba(64,65,192,.35)] cursor-pointer hover:bg-primary-800'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
          <p className="mt-1.5 text-[11px] text-gray-400 text-center">Enter to send &nbsp;·&nbsp; Shift+Enter for new line</p>
        </div>
      </footer>
    </div>
  )
}

export default Chat
