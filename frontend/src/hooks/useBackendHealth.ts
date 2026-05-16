/**
 * useBackendHealth — polls GET /api/health every POLL_MS milliseconds.
 *
 * Returns one of three statuses:
 *   "mock"    — VITE_USE_MOCK is not 'false'; no real backend expected
 *   "online"  — last health check succeeded
 *   "offline" — last health check failed (network error or non-2xx)
 *   "pending" — first check has not completed yet
 */

import { useEffect, useRef, useState } from 'react'

export type BackendStatus = 'mock' | 'online' | 'offline' | 'pending'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'
const POLL_MS = 15_000

async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch('/api/health', { signal: AbortSignal.timeout(5_000) })
    return res.ok
  } catch {
    return false
  }
}

export function useBackendHealth(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>(USE_MOCK ? 'mock' : 'pending')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (USE_MOCK) return

    let cancelled = false

    async function poll() {
      if (cancelled) return
      const ok = await checkHealth()
      if (!cancelled) setStatus(ok ? 'online' : 'offline')
      if (!cancelled) timerRef.current = setTimeout(poll, POLL_MS)
    }

    poll()

    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  return status
}
