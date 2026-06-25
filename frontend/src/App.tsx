import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { HealthContext } from './context/HealthContext'
import useHealth from './hooks/useHealth'
import Header from './components/Header'
import Chat from './pages/Chat'
import Ingest from './pages/Ingest'

const AppShell = () => {
  const { status, retry, nextCheckIn } = useHealth()

  return (
    <HealthContext.Provider value={{ status, retry, nextCheckIn }}>
      <div className="flex flex-col flex-1 min-h-0 bg-gray-50">
        <Header />
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/ingest" element={<Ingest />} />
        </Routes>
      </div>
    </HealthContext.Provider>
  )
}

const App = () => (
  <BrowserRouter>
    <AppShell />
  </BrowserRouter>
)

export default App
