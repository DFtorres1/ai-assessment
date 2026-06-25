export type UserType = 'member' | 'staff'

export type HealthStatus = 'checking' | 'ok' | 'degraded' | 'offline' | 'no_api_key'

export interface Citation {
  doc_name: string
  page: number
  section: string
}

export interface ToolCall {
  tool: string
  input: Record<string, unknown>
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations: Citation[]
  tool_calls: ToolCall[]
  streaming: boolean
  error?: string
}
