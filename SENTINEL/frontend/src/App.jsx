import { useState, useEffect } from 'react'
import { Sidebar } from './components/Sidebar'
import { StatusBar } from './components/StatusBar'
import Dashboard from './pages/Dashboard'
import Classify from './pages/Classify'
import Training from './pages/Training'
import Analytics from './pages/Analytics'

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [status, setStatus] = useState({
    model_loaded: false,
    accuracy: 0,
    features_loaded: false,
    dataset_stats: null,
  })
  const [logs, setLogs] = useState([])

  const addLog = (message, level = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev.slice(-50), { time, message, level }])
  }

  const refreshStatus = async () => {
    try {
      const res = await fetch('/api/status')
      const data = await res.json()
      setStatus(data)
    } catch {
      addLog('Failed to connect to server', 'error')
    }
  }

  useEffect(() => {
    refreshStatus()
    addLog('SENTINEL system initialized')
    addLog('Connecting to backend server...')
  }, [])

  const renderPage = () => {
    const props = { status, refreshStatus, addLog, logs }
    switch (page) {
      case 'dashboard': return <Dashboard {...props} />
      case 'classify': return <Classify {...props} />
      case 'training': return <Training {...props} />
      case 'analytics': return <Analytics {...props} />
      default: return <Dashboard {...props} />
    }
  }

  return (
    <>
      <StatusBar status={status} />
      <div className="app-layout">
        <Sidebar activePage={page} onNavigate={setPage} />
        <main className="main-content">
          <div className="page-enter" key={page}>
            {renderPage()}
          </div>
        </main>
      </div>
    </>
  )
}
