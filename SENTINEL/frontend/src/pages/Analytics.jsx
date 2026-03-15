import { useState, useEffect } from 'react'
import { GlowCard } from '../components/Components'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function Analytics({ addLog }) {
    const [history, setHistory] = useState(null)
    const [metrics, setMetrics] = useState(null)
    const [loading, setLoading] = useState(true)
    const [modelInfo, setModelInfo] = useState(null)

    useEffect(() => {
        loadAll()
    }, [])

    const loadAll = async () => {
        setLoading(true)
        await Promise.all([loadHistory(), loadMetrics(), loadModelInfo()])
        setLoading(false)
    }

    const loadHistory = async () => {
        try {
            const res = await fetch('/api/history')
            const data = await res.json()
            if (data.success) {
                setHistory(data.history)
                addLog('Training history loaded')
            }
        } catch {
            addLog('Failed to load history', 'error')
        }
    }

    const loadMetrics = async () => {
        try {
            const res = await fetch('/api/metrics')
            const data = await res.json()
            if (data.success) {
                setMetrics(data)
                addLog('Classification metrics loaded')
            }
        } catch { }
    }

    const loadModelInfo = async () => {
        try {
            const res = await fetch('/api/model/info')
            const data = await res.json()
            if (data.success) setModelInfo(data.info)
        } catch { }
    }

    const hasVal = history && history.val_accuracy

    const chartData = history ? history.accuracy.map((acc, i) => ({
        epoch: i + 1,
        accuracy: (acc * 100).toFixed(2),
        loss: history.loss[i].toFixed(4),
        ...(hasVal ? {
            val_accuracy: (history.val_accuracy[i] * 100).toFixed(2),
            val_loss: history.val_loss[i].toFixed(4),
        } : {}),
    })) : []

    const finalAcc = history ? (history.accuracy[history.accuracy.length - 1] * 100).toFixed(2) : '—'
    const finalValAcc = hasVal ? (history.val_accuracy[history.val_accuracy.length - 1] * 100).toFixed(2) : '—'
    const finalLoss = history ? history.loss[history.loss.length - 1].toFixed(4) : '—'
    const epochsRun = history ? history.accuracy.length : '—'

    const customTooltip = ({ active, payload, label }) => {
        if (!active || !payload) return null
        return (
            <div style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '12px 16px',
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
            }}>
                <div style={{ color: 'var(--text-dim)', marginBottom: 8 }}>EPOCH {label}</div>
                {payload.map(p => (
                    <div key={p.name} style={{ color: p.color, marginBottom: 4 }}>
                        {p.name.toUpperCase()}: {p.value}
                    </div>
                ))}
            </div>
        )
    }

    return (
        <div>
            <div className="page-header">
                <h1 className="page-header__title">◈ ANALYTICS</h1>
                <p className="page-header__subtitle">Training performance metrics and model analysis</p>
            </div>

            {/* Summary Cards */}
            <div className="dashboard-grid mb-24">
                <GlowCard title="Training Accuracy">
                    <div className="stat-value">{finalAcc}%</div>
                    <div className="stat-label">Final Training Accuracy</div>
                </GlowCard>
                <GlowCard title="Validation Accuracy">
                    <div className="stat-value">{finalValAcc}%</div>
                    <div className="stat-label">Generalization Score</div>
                </GlowCard>
                <GlowCard title="Final Loss">
                    <div className="stat-value">{finalLoss}</div>
                    <div className="stat-label">Cross-Entropy Loss</div>
                </GlowCard>
                <GlowCard title="Epochs Run">
                    <div className="stat-value">{epochsRun}</div>
                    <div className="stat-label">{modelInfo ? `${modelInfo.version} Model` : 'Epochs'}</div>
                </GlowCard>
            </div>

            {/* Model Info */}
            {modelInfo && (
                <GlowCard title="Model Architecture" className="mb-24">
                    <div className="dashboard-grid">
                        <div>
                            <div className="stat-value" style={{ fontSize: 22 }}>{modelInfo.version.toUpperCase()}</div>
                            <div className="stat-label">Model Version</div>
                        </div>
                        <div>
                            <div className="stat-value" style={{ fontSize: 22 }}>{(modelInfo.total_params / 1000).toFixed(0)}K</div>
                            <div className="stat-label">Parameters</div>
                        </div>
                        <div>
                            <div className="stat-value" style={{ fontSize: 22 }}>{modelInfo.num_layers}</div>
                            <div className="stat-label">Layers</div>
                        </div>
                    </div>
                </GlowCard>
            )}

            {/* Charts */}
            {!loading && history && (
                <>
                    <GlowCard title="Accuracy Over Epochs" className="mb-24">
                        <div style={{ width: '100%', height: 300 }}>
                            <ResponsiveContainer>
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                    <XAxis dataKey="epoch" stroke="var(--text-dim)" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <YAxis stroke="var(--text-dim)" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <Tooltip content={customTooltip} />
                                    <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <Line type="monotone" dataKey="accuracy" stroke="#00ff88" strokeWidth={2} dot={{ r: 2 }} name="Train Acc %" />
                                    {hasVal && <Line type="monotone" dataKey="val_accuracy" stroke="#00bbff" strokeWidth={2} dot={{ r: 2 }} name="Val Acc %" strokeDasharray="5 5" />}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </GlowCard>

                    <GlowCard title="Loss Over Epochs" className="mb-24">
                        <div style={{ width: '100%', height: 300 }}>
                            <ResponsiveContainer>
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                    <XAxis dataKey="epoch" stroke="var(--text-dim)" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <YAxis stroke="var(--text-dim)" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <Tooltip content={customTooltip} />
                                    <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
                                    <Line type="monotone" dataKey="loss" stroke="#ffaa00" strokeWidth={2} dot={{ r: 2 }} name="Train Loss" />
                                    {hasVal && <Line type="monotone" dataKey="val_loss" stroke="#ff6644" strokeWidth={2} dot={{ r: 2 }} name="Val Loss" strokeDasharray="5 5" />}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </GlowCard>
                </>
            )}

            {/* Confusion Matrix */}
            {metrics && metrics.confusion_matrix && (
                <GlowCard title="Confusion Matrix" className="mb-24">
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                            <thead>
                                <tr>
                                    <th style={{ padding: '10px 12px', textAlign: 'left', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)' }}>
                                        Actual ↓ / Predicted →
                                    </th>
                                    {metrics.labels.map(label => (
                                        <th key={label} style={{ padding: '10px 12px', textAlign: 'center', color: 'var(--accent)', borderBottom: '1px solid var(--border)', fontSize: 10 }}>
                                            {label}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {metrics.confusion_matrix.map((row, i) => (
                                    <tr key={i}>
                                        <td style={{ padding: '10px 12px', color: 'var(--accent)', borderBottom: '1px solid var(--border)', fontSize: 11 }}>
                                            {metrics.labels[i]}
                                        </td>
                                        {row.map((val, j) => {
                                            const maxInRow = Math.max(...row)
                                            const intensity = maxInRow > 0 ? val / maxInRow : 0
                                            const isCorrect = i === j
                                            return (
                                                <td key={j} style={{
                                                    padding: '10px 12px',
                                                    textAlign: 'center',
                                                    borderBottom: '1px solid var(--border)',
                                                    background: isCorrect
                                                        ? `rgba(0, 255, 136, ${intensity * 0.3})`
                                                        : val > 0 ? `rgba(255, 51, 102, ${intensity * 0.2})` : 'transparent',
                                                    color: val > 0 ? 'var(--text-primary)' : 'var(--text-dim)',
                                                    fontWeight: isCorrect ? 700 : 400,
                                                }}>
                                                    {val}
                                                </td>
                                            )
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </GlowCard>
            )}

            {/* Per-Class Metrics */}
            {metrics && metrics.classification_report && (
                <GlowCard title="Per-Class Performance">
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                            <thead>
                                <tr>
                                    {['Class', 'Precision', 'Recall', 'F1-Score', 'Support'].map(h => (
                                        <th key={h} style={{ padding: '10px 12px', textAlign: h === 'Class' ? 'left' : 'center', color: 'var(--accent)', borderBottom: '1px solid var(--border)' }}>
                                            {h}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {metrics.labels.map(label => {
                                    const r = metrics.classification_report[label]
                                    if (!r) return null
                                    return (
                                        <tr key={label}>
                                            <td style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', color: 'var(--text-primary)' }}>{label}</td>
                                            <td style={{ padding: '10px 12px', textAlign: 'center', borderBottom: '1px solid var(--border)', color: r.precision > 0.8 ? 'var(--accent)' : 'var(--accent-amber)' }}>
                                                {(r.precision * 100).toFixed(1)}%
                                            </td>
                                            <td style={{ padding: '10px 12px', textAlign: 'center', borderBottom: '1px solid var(--border)', color: r.recall > 0.8 ? 'var(--accent)' : 'var(--accent-amber)' }}>
                                                {(r.recall * 100).toFixed(1)}%
                                            </td>
                                            <td style={{ padding: '10px 12px', textAlign: 'center', borderBottom: '1px solid var(--border)', color: r['f1-score'] > 0.8 ? 'var(--accent)' : 'var(--accent-amber)' }}>
                                                {(r['f1-score'] * 100).toFixed(1)}%
                                            </td>
                                            <td style={{ padding: '10px 12px', textAlign: 'center', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                                                {r.support}
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                </GlowCard>
            )}

            {/* Empty States */}
            {!loading && !history && (
                <GlowCard title="No Data">
                    <div style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                        &gt; No training history available<br />
                        &gt; Train the model to view analytics
                    </div>
                </GlowCard>
            )}

            {loading && (
                <GlowCard title="Loading">
                    <div style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                        <span className="tactical-btn__spinner" style={{ display: 'inline-block' }} /> Loading analytics data...
                    </div>
                </GlowCard>
            )}
        </div>
    )
}
