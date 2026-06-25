import { NavLink } from 'react-router-dom'
import BlossomMark from './BlossomMark'
import { useHealthContext } from '../context/HealthContext'

const Header = () => {
  const { status } = useHealthContext()

  const dotClass =
    status === 'ok' ? 'bg-green-400' :
      status === 'checking' ? 'bg-yellow-400 animate-pulse' :
        'bg-red-400'

  const statusLabel =
    status === 'checking' ? 'Connecting…' :
      status === 'ok' ? 'Online' :
        status === 'no_api_key' ? 'No API Key' :
          status === 'degraded' ? 'Degraded' : 'Offline'

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm flex-shrink-0">
      <div className="max-w-4xl mx-auto px-5 py-2.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BlossomMark size={36} className="text-primary-700" />
          <div>
            <div className="font-bold text-[15px] text-gray-900 leading-tight">Blossom Banking</div>
            <div className="text-xs text-gray-500 mt-0.5">Login &amp; Security Helper</div>
          </div>
        </div>

        <nav className="flex items-center gap-5">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-primary-700' : 'text-gray-500 hover:text-gray-800'}`
            }
          >
            Chat
          </NavLink>
          <NavLink
            to="/ingest"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${isActive ? 'text-primary-700' : 'text-gray-500 hover:text-gray-800'}`
            }
          >
            Ingest
          </NavLink>
          <div className="flex items-center gap-1.5 text-xs text-gray-400 ml-1">
            <span className={`w-2 h-2 rounded-full ${dotClass}`} />
            {statusLabel}
          </div>
        </nav>
      </div>
    </header>
  )
}

export default Header