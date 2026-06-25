import { useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

interface IngestResult {
  status: 'success' | 'error'
  message: string
  detail?: string
}

const Ingest = () => {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    setIsLoading(true)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json() as Record<string, unknown>

      if (!res.ok) {
        setResult({ status: 'error', message: `Error ${res.status}`, detail: (data.detail as string) ?? JSON.stringify(data) })
      } else {
        setResult({ status: 'success', message: 'Document ingested successfully.', detail: JSON.stringify(data, null, 2) })
      }
    } catch (err) {
      setResult({ status: 'error', message: 'Network error', detail: String(err) })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-xl mx-auto px-5 py-10">
        <h1 className="text-xl font-bold text-gray-900 mb-1">Ingest Document</h1>
        <p className="text-sm text-gray-500 mb-8">Upload a PDF or text file to add it to the knowledge base.</p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5" htmlFor="file-upload">
              Document
            </label>
            <div className="border-2 border-dashed border-gray-200 rounded-2xl p-6 text-center hover:border-primary-300 transition-colors">
              <input
                id="file-upload"
                type="file"
                accept=".pdf,.txt,.md"
                onChange={e => setFile(e.target.files?.[0] ?? null)}
                className="hidden"
              />
              <label htmlFor="file-upload" className="cursor-pointer block">
                {file ? (
                  <div>
                    <p className="font-medium text-gray-800">{file.name}</p>
                    <p className="text-xs text-gray-400 mt-1">{(file.size / 1024).toFixed(1)} KB · Click to change</p>
                  </div>
                ) : (
                  <div>
                    <svg className="mx-auto mb-2 text-gray-300" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <p className="text-sm text-gray-500">Click to choose a file</p>
                    <p className="text-xs text-gray-400 mt-0.5">PDF, TXT, or MD</p>
                  </div>
                )}
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={!file || isLoading}
            className="w-full py-3 rounded-xl bg-primary-700 text-white text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary-800 shadow-[0_2px_8px_rgba(64,65,192,.3)] border-none cursor-pointer"
          >
            {isLoading ? 'Uploading…' : 'Ingest Document'}
          </button>
        </form>

        {result && (
          <div
            className={`mt-6 p-4 rounded-2xl border text-sm ${
              result.status === 'success'
                ? 'bg-green-50 border-green-200 text-green-800'
                : 'bg-red-50 border-red-200 text-red-800'
            }`}
          >
            <p className="font-semibold">{result.message}</p>
            {result.detail && (
              <pre className="mt-2 text-xs overflow-x-auto whitespace-pre-wrap font-mono">{result.detail}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Ingest
