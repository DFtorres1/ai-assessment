import { createContext, useContext } from 'react'
import type { HealthStatus } from '../types'

interface HealthContextValue {
  status: HealthStatus
  retry: () => void
  nextCheckIn: number
}

export const HealthContext = createContext<HealthContextValue>({
  status: 'checking',
  retry: () => {},
  nextCheckIn: 30,
})

export const useHealthContext = () => useContext(HealthContext)
