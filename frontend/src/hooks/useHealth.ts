import { useCallback, useEffect, useRef, useState } from 'react'
import type { HealthStatus } from '../types'

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''
const POLL_MS = 10_000

const useHealth = () => {
  const [status, setStatus] = useState<HealthStatus>('checking')
  const [nextCheckIn, setNextCheckIn] = useState(POLL_MS / 1000)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const runCheck = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (!res.ok) { setStatus('degraded'); return }
      const data = await res.json() as { status?: string; checks?: Record<string, string> }
      if (data.checks?.anthropic_api_key === 'missing') {
        setStatus('no_api_key')
      } else {
        setStatus(data.status === 'ok' ? 'ok' : 'degraded')
      }
    } catch {
      setStatus('offline')
    }
  }, [])

  const startCycle = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    if (tickRef.current) clearInterval(tickRef.current)

    setNextCheckIn(POLL_MS / 1000)

    tickRef.current = setInterval(() => {
      setNextCheckIn(s => Math.max(0, s - 1))
    }, 1000)

    pollRef.current = setInterval(() => {
      runCheck()
      setNextCheckIn(POLL_MS / 1000)
    }, POLL_MS)
  }, [runCheck])

  const retry = useCallback(() => {
    runCheck()
    startCycle()
  }, [runCheck, startCycle])

  useEffect(() => {
    runCheck()
    startCycle()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      if (tickRef.current) clearInterval(tickRef.current)
    }
  }, [runCheck, startCycle])

  return { status, retry, nextCheckIn }
}

export default useHealth
