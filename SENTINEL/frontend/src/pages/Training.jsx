import { useState } from 'react'
import { GlowCard, LogConsole } from '../components/Components'

export default function Training({ status, refreshStatus, addLog, logs }) {
    const [epochs, setEpochs] = useState(20)
    const [training, setTraining] = useState(false)
    const [augmentation, setAugmentation] = useState(true)
    const [progress, setProgress] = useState(0)
    const [trainResult, setTrainResult] = useState(null)

    const startTraining = async () => {
        setTraining(true)
        setProgress(0)
        setTrainResult(null)
        addLog(`Starting CNN training — ${epochs} epochs, augmentation=${augmentation}`)

        // Simulate progress while training
        const interval = setInterval(() => {
            setProgress(prev => Math.min(prev + (100 / epochs / 2), 95))
        }, 1000)

        try {
            const res = await fetch('/api/model/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs, augmentation }),
            })
            const data = await res.json()
            clearInterval(interval)

            if (data.success) {
                setProgress(100)
                setTrainResult(data)
                addLog(`Training complete — train: ${data.accuracy}%, val: ${data.val_accuracy}%`, 'info')
                await refreshStatus()
            } else {
                addLog(`Training failed: ${data.error}`, 'error')
            }
        } catch {
            clearInterval(interval)
            addLog('Training failed — server error', 'error')
        }
        setTraining(false)
    }

    const loadModel = async () => {
        addLog('Loading pre-trained model...')
        try {
            await fetch('/api/features/load', { method: 'POST' })
            const res = await fetch('/api/model/load', { method: 'POST' })
            const data = await res.json()
            if (data.success) {
                addLog(`Model loaded — accuracy: ${data.accuracy}%`, 'info')
                await refreshStatus()
            } else {
                addLog(`Load failed: ${data.error}`, 'error')
            }
        } catch {
            addLog('Server error', 'error')
        }
    }

    return (
        <div>
            <div className="page-header">
                <h1 className="page-header__title">◈ MODEL TRAINING</h1>
                <p className="page-header__subtitle">Train, configure, or load the CNN classification model</p>
            </div>

            <div className="dashboard-grid">
                <GlowCard title="Current Model">
                    <div className="stat-value">{status.model_loaded ? `${status.accuracy}%` : 'NOT LOADED'}</div>
                    <div className="stat-label mb-16">
                        {status.model_loaded
                            ? `Train ACC | ${status.model_info?.version || 'legacy'} model`
                            : 'Load or train to begin'}
                    </div>
                    {status.val_accuracy > 0 && (
                        <div style={{ marginBottom: 16 }}>
                            <span className="text-mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                                Val Accuracy: <span className="text-accent">{status.val_accuracy}%</span>
                            </span>
                        </div>
                    )}
                    <button className="tactical-btn" onClick={loadModel} disabled={training} style={{ width: '100%' }}>
                        LOAD PRE-TRAINED WEIGHTS
                    </button>
                </GlowCard>

                <GlowCard title="Train New Model">
                    <div className="training-controls">
                        <div>
                            <div className="stat-label mb-16">EPOCHS</div>
                            <input
                                type="number"
                                className="training-input"
                                value={epochs}
                                onChange={(e) => setEpochs(parseInt(e.target.value) || 20)}
                                min={1}
                                max={100}
                                disabled={training}
                            />
                        </div>
                        <div>
                            <div className="stat-label mb-16">AUGMENTATION</div>
                            <button
                                className={`tactical-btn ${augmentation ? 'tactical-btn--primary' : ''}`}
                                onClick={() => setAugmentation(!augmentation)}
                                disabled={training}
                                style={{ padding: '10px 16px', fontSize: 11 }}
                            >
                                {augmentation ? 'ON ✓' : 'OFF'}
                            </button>
                        </div>
                    </div>
                    <button className="tactical-btn tactical-btn--primary" onClick={startTraining} disabled={training} style={{ width: '100%' }}>
                        {training ? <><span className="tactical-btn__spinner" />TRAINING IN PROGRESS...</> : 'START TRAINING'}
                    </button>
                    <div className="text-dim text-mono mt-16" style={{ fontSize: 10, lineHeight: 1.6 }}>
                        Enhanced CNN v2 • BatchNorm • Dropout • Early Stopping
                        {augmentation && ' • Data Augmentation'}
                    </div>
                </GlowCard>
            </div>

            {(training || progress > 0) && (
                <GlowCard title="Training Progress" className="mb-24">
                    <div className="flex items-center gap-16 mb-16">
                        <span className="text-mono" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                            {progress < 100 ? `Training... ${progress.toFixed(0)}%` : 'Training Complete ✓'}
                        </span>
                    </div>
                    <div className="progress-bar">
                        <div className="progress-bar__fill" style={{ width: `${progress}%` }} />
                    </div>
                </GlowCard>
            )}

            {trainResult && trainResult.history && (
                <GlowCard title="Training Results" className="mb-24">
                    <div className="dashboard-grid">
                        <div>
                            <div className="stat-value">{trainResult.accuracy}%</div>
                            <div className="stat-label">Train Accuracy</div>
                        </div>
                        <div>
                            <div className="stat-value">{trainResult.val_accuracy}%</div>
                            <div className="stat-label">Val Accuracy</div>
                        </div>
                        <div>
                            <div className="stat-value">{trainResult.history.loss[trainResult.history.loss.length - 1].toFixed(4)}</div>
                            <div className="stat-label">Final Loss</div>
                        </div>
                        <div>
                            <div className="stat-value">{trainResult.history.accuracy.length}</div>
                            <div className="stat-label">Epochs Run</div>
                        </div>
                    </div>
                </GlowCard>
            )}

            <GlowCard title="Training Log">
                <LogConsole logs={logs} />
            </GlowCard>
        </div>
    )
}
