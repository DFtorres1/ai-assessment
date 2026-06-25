import { useHealthContext } from '../context/HealthContext'

const HealthBanner = () => {
  const { status, retry, nextCheckIn } = useHealthContext()

  if (status === 'no_api_key') {
    return (
      <div className="bg-red-50 border-b-2 border-red-400 px-6 py-5 flex-shrink-0" role="alert">
        <div className="max-w-3xl mx-auto flex items-start gap-4">
          <span className="text-red-500 text-3xl leading-none flex-shrink-0" aria-hidden="true">⚠</span>
          <div className="flex-1">
            <p className="font-bold text-red-800 text-base mb-1">
              ANTHROPIC_API_KEY is not configured — AI features are disabled
            </p>
            <p className="text-red-700 text-sm mb-3">
              The assistant cannot answer questions without a valid Anthropic API key.
              Set the key and restart the backend to restore full functionality.
            </p>
            <code className="block bg-red-100 border border-red-300 rounded-lg px-4 py-2.5 text-sm text-red-900 font-mono leading-relaxed">
              export ANTHROPIC_API_KEY=sk-ant-…<br />
              docker compose restart backend
            </code>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'ok' || status === 'checking') return null

  return (
    <div className="bg-amber-50 border-b border-amber-300 px-5 py-2 flex items-center justify-center gap-4 text-sm text-amber-800 flex-shrink-0">
      <span>
        {status === 'offline' ? 'Backend is offline.' : 'Backend is degraded.'}
        {' '}Retrying in {nextCheckIn}s
      </span>
      <button
        onClick={retry}
        className="px-3 py-1 rounded-lg border border-amber-400 bg-amber-100 text-amber-900 text-xs font-semibold hover:bg-amber-200 transition-colors cursor-pointer"
      >
        Retry now
      </button>
    </div>
  )
}

export default HealthBanner
