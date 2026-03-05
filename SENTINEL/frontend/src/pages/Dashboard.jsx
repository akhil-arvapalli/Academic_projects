import { useState, useEffect } from 'react'
import { GlowCard, LogConsole } from '../components/Components'
import { Satellite, Database, Brain, Activity } from 'lucide-react'

export default function Dashboard({ status, refreshStatus, addLog, logs }) {
    const [initializing, setInitializing] = useState(false)

    const initializeSystem = async () => {
        setInitializing(true)
        addLog('Loading feature dataset...')
        try {
            const featRes = await fetch('/api/features/load', { method: 'POST' })
            const featData = await featRes.json()
            if (featData.success) {
                addLog(`Features loaded: ${featData.stats.total_images} images`, 'info')
            } else {
                addLog(`Feature loading failed: ${featData.error}`, 'error')
            }
        } catch {
            addLog('Server connection failed', 'error')
            setInitializing(false)
            return
        }

        addLog('Loading CNN model weights...')
        try {
            const modelRes = await fetch('/api/model/load', { method: 'POST' })
            const modelData = await modelRes.json()
            if (modelData.success) {
                addLog(`Model loaded — accuracy: ${modelData.accuracy}%`, 'info')
            } else {
                addLog(`Model loading failed: ${modelData.error}`, 'warn')
            }
        } catch {
            addLog('Model loading failed', 'error')
        }

        await refreshStatus()
        setInitializing(false)
        addLog('System ready ✓')
    }

    const stats = status.dataset_stats

    return (
        <div>
            <div className="page-header">
                <h1 className="page-header__title">◈ COMMAND CENTER</h1>
                <p className="page-header__subtitle">System overview and operational status</p>
            </div>

            <div className="dashboard-grid">
                <GlowCard title="Model Status">
                    <div className="flex items-center gap-12" style={{ marginBottom: 12 }}>
                        <Brain size={24} style={{ color: status.model_loaded ? 'var(--accent)' : 'var(--text-dim)' }} />
                        <div>
                            <div className="stat-value">{status.model_loaded ? 'ONLINE' : 'OFFLINE'}</div>
                            <div className="stat-label">CNN Classifier</div>
                        </div>
                    </div>
                    {!status.model_loaded && (
                        <button className="tactical-btn tactical-btn--primary" onClick={initializeSystem} disabled={initializing} style={{ marginTop: 8, width: '100%' }}>
                            {initializing && <span className="tactical-btn__spinner" />}
                            {initializing ? 'INITIALIZING...' : 'INITIALIZE SYSTEM'}
                        </button>
                    )}
                </GlowCard>

                <GlowCard title="Accuracy">
                    <div className="flex items-center gap-12">
                        <Activity size={24} style={{ color: 'var(--accent)' }} />
                        <div>
                            <div className="stat-value">{status.accuracy > 0 ? `${status.accuracy}%` : '—'}</div>
                            <div className="stat-label">Classification Accuracy</div>
                        </div>
                    </div>
                </GlowCard>

                <GlowCard title="Dataset">
                    <div className="flex items-center gap-12">
                        <Database size={24} style={{ color: status.features_loaded ? 'var(--accent)' : 'var(--text-dim)' }} />
                        <div>
                            <div className="stat-value">{stats ? stats.total_images : '—'}</div>
                            <div className="stat-label">Satellite Images</div>
                        </div>
                    </div>
                </GlowCard>

                <GlowCard title="Classification Classes">
                    <div className="flex items-center gap-12">
                        <Satellite size={24} style={{ color: 'var(--accent)' }} />
                        <div>
                            <div className="stat-value">4</div>
                            <div className="stat-label">Land Types</div>
                        </div>
                    </div>
                </GlowCard>
            </div>

            {stats && stats.class_distribution && (
                <GlowCard title="Class Distribution" className="mb-24">
                    <div className="flex flex-col gap-12">
                        {Object.entries(stats.class_distribution).map(([label, count]) => {
                            const pct = (count / stats.total_images * 100).toFixed(1)
                            const icons = { 'Urban Land': '🏙️', 'Agricultural Land': '🌾', 'Range Land': '🏜️', 'Forest Land': '🌲' }
                            return (
                                <div key={label} className="confidence-bar">
                                    <div className="confidence-bar__header">
                                        <span className="confidence-bar__label">{icons[label]} {label}</span>
                                        <span className="confidence-bar__value">{count} ({pct}%)</span>
                                    </div>
                                    <div className="confidence-bar__track">
                                        <div className="confidence-bar__fill" style={{ width: `${pct}%` }} />
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </GlowCard>
            )}

            <GlowCard title="System Log">
                <LogConsole logs={logs} />
            </GlowCard>
        </div>
    )
}
