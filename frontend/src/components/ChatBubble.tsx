import type { Message } from '../types'
import BlossomMark from './BlossomMark'
import CitationList from './CitationList'
import HolidayBadge from './HolidayBadge'

const TypingDots = () => (
  <div className="flex gap-1 py-0.5">
    <span className="dot" />
    <span className="dot" />
    <span className="dot" />
  </div>
)

interface ChatBubbleProps {
  msg: Message
}

const ChatBubble = ({ msg }: ChatBubbleProps) => {
  const isUser = msg.role === 'user'

  return (
    <div className={`msg-enter flex items-end gap-2.5 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 leading-none">
          <BlossomMark size={28} className="text-primary-600" />
        </div>
      )}
      <div className="max-w-[72%] min-w-0">
        <div
          className={
            isUser
              ? 'px-4 py-[11px] rounded-[20px] rounded-br-[5px] text-[15px] leading-relaxed whitespace-pre-wrap break-words bg-primary-700 text-white'
              : 'px-4 py-[11px] rounded-[20px] rounded-bl-[5px] text-[15px] leading-relaxed whitespace-pre-wrap break-words bg-white text-gray-900 shadow-[0_1px_4px_rgba(64,65,192,.08),0_0_0_1px_rgba(209,213,219,.5)]'
          }
        >
          {msg.error ? (
            <span className="text-red-500 text-sm">{msg.error}</span>
          ) : msg.text ? (
            <>
              {msg.text}
              {msg.streaming && (
                <span className="inline-block w-0.5 h-[1.1em] bg-primary-400 ml-0.5 align-text-bottom animate-[blink_1s_step-start_infinite]" />
              )}
            </>
          ) : msg.streaming ? (
            <TypingDots />
          ) : null}
        </div>
        {!isUser && <HolidayBadge tool_calls={msg.tool_calls} />}
        {!isUser && !msg.streaming && <CitationList citations={msg.citations} />}
      </div>
    </div>
  )
}

export default ChatBubble
