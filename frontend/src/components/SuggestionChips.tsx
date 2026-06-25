interface SuggestionChipsProps {
  onSelect: (text: string) => void
}

const SUGGESTIONS = [
  { label: 'Reset my password', icon: '🔑' },
  { label: 'Account locked out', icon: '🔒' },
  { label: 'MFA code not working', icon: '📱' },
  { label: 'Trusted device not recognized', icon: '💻' },
]

const SuggestionChips = ({ onSelect }: SuggestionChipsProps) => (
  <div className="flex flex-col items-center gap-6 py-12 text-center">
    <div>
      <p className="text-base font-semibold text-gray-700">How can I help you today?</p>
      <p className="text-sm text-gray-400 mt-1">
        Ask me anything about login issues, passwords, MFA, or device recognition — or pick a topic below.
      </p>
    </div>
    <div className="grid grid-cols-2 gap-3 w-full max-w-sm">
      {SUGGESTIONS.map(s => (
        <button
          key={s.label}
          onClick={() => onSelect(s.label)}
          className="flex items-center gap-2 px-4 py-3 bg-white border border-gray-200 rounded-2xl text-sm font-medium text-gray-700 text-left hover:border-primary-400 hover:text-primary-700 transition-colors shadow-sm cursor-pointer"
        >
          <span aria-hidden>{s.icon}</span>
          {s.label}
        </button>
      ))}
    </div>
  </div>
)

export default SuggestionChips
