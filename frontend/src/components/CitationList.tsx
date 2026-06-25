import { useState } from 'react'
import type { Citation } from '../types'

interface CitationListProps {
  citations: Citation[]
}

const CitationList = ({ citations }: CitationListProps) => {
  const [open, setOpen] = useState(false)

  if (!citations.length) return null

  return (
    <div className="mt-2 text-sm">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-[11px] font-bold text-primary-700 uppercase tracking-wider px-1 py-0.5 rounded hover:bg-primary-50 transition-colors"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="flex-shrink-0">
          <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Sources ({citations.length})
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
          className={`flex-shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="mt-1 p-3 bg-primary-50 border border-primary-200 rounded-2xl">
          {citations.map((c, i) => (
            <div key={i} className="flex items-baseline flex-wrap gap-1.5 mb-1 last:mb-0">
              <span className="font-semibold text-gray-800">{c.doc_name}</span>
              <span className="text-xs text-gray-500">p.{c.page}{c.section ? ` · ${c.section}` : ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default CitationList
