import { useState } from 'react'
import { GlowCard, LogConsole } from '../components/Components'

export default function Training({ status, refreshStatus, addLog, logs }) {
    const [epochsInput, setEpochsInput] = useState('20')
    const [training, setTraining] = useState(false)
    const [augmentation, setAugmentation] = useState(true)
    const [modelType, setModelType] = useState('cnn')
    const [progress, setProgress] = useState(0)
    const [trainResult, setTrainResult] = useState(null)

    const clampEpochs = (value) => {
        if (!Number.isFinite(value)) return 20
        return Math.min(200, Math.max(1, Math.round(value)))
    }

    const getEpochs = () => {
        const parsed = Number.parseInt(epochsInput, 10)
        return clampEpochs(parsed)
    }

    const applyEpochs = (next) => {
        setEpochsInput(String(clampEpochs(next)))
    }

    const startTraining = async () => {
        const epochs = getEpochs()
        // Normalize any partially typed value before training starts.
        applyEpochs(epochs)
        setTraining(true)
        setProgress(0)
        setTrainResult(null)
        addLog(`Starting ${modelType.toUpperCase()} training — ${epochs} epochs, augmentation=${augmentation}`)

        // Simulate progress while training
        const interval = setInterval(() => {
            setProgress(prev => Math.min(prev + (100 / epochs / 2), 95))
        }, 1000)

        try {
            const res = await fetch('/api/model/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs, augmentation, model_type: modelType }),
            })
            const data = await res.json()
            clearInterval(interval)

            if (data.success) {
                setProgress(100)
                setTrainResult(data)
                addLog(
                    `Training complete (${(data.model_type || modelType).toUpperCase()}) — train: ${data.accuracy}%, val: ${data.val_accuracy}%`,
                    'info'
                )
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
        addLog(`Loading pre-trained model (${modelType.toUpperCase()})...`)
        try {
            await fetch('/api/features/load', { method: 'POST' })
            const res = await fetch('/api/model/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType }),
            })
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
                <p className="page-header__subtitle">Train, configure, or load CNN/EfficientNet classification models</p>
            </div>

            <div className="dashboard-grid">
                <GlowCard title="Current Model">
                    <div className="stat-value">{status.model_loaded ? `${status.accuracy}%` : 'NOT LOADED'}</div>
                    <div className="stat-label mb-16">
                        {status.model_loaded
                            ? `${(status.model_info?.model_type || 'cnn').toUpperCase()} | ${status.model_info?.version || 'legacy'} model`
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
                            <div className="stat-label mb-16">MODEL TYPE</div>
                            <div style={{ display: 'flex', gap: 8 }}>
                                <button
                                    className={`tactical-btn ${modelType === 'cnn' ? 'tactical-btn--primary' : ''}`}
                                    onClick={() => setModelType('cnn')}
                                    disabled={training}
                                    style={{ padding: '10px 12px', fontSize: 11 }}
                                >
                                    CNN
                                </button>
                                <button
                                    className={`tactical-btn ${modelType === 'efficientnet' ? 'tactical-btn--primary' : ''}`}
                                    onClick={() => setModelType('efficientnet')}
                                    disabled={training}
                                    style={{ padding: '10px 12px', fontSize: 11 }}
                                >
                                    EFFICIENTNET
                                </button>
                            </div>
                        </div>
                        <div>
                            <div className="stat-label mb-16">EPOCHS</div>
                            <div className="epochs-control">
                                <button
                                    className="tactical-btn epochs-btn"
                                    type="button"
                                    onClick={() => applyEpochs(getEpochs() - 1)}
                                    disabled={training}
                                >
                                    -
                                </button>
                                <input
                                    type="text"
                                    inputMode="numeric"
                                    pattern="[0-9]*"
                                    className="training-input"
                                    value={epochsInput}
                                    onChange={(e) => {
                                        const digitsOnly = e.target.value.replace(/[^0-9]/g, '')
                                        setEpochsInput(digitsOnly)
                                    }}
                                    onBlur={() => applyEpochs(getEpochs())}
                                    disabled={training}
                                    aria-label="Epochs"
                                />
                                <button
                                    className="tactical-btn epochs-btn"
                                    type="button"
                                    onClick={() => applyEpochs(getEpochs() + 1)}
                                    disabled={training}
                                >
                                    +
                                </button>
                            </div>
                            <div className="epochs-presets">
                                {[10, 20, 30, 50].map((preset) => (
                                    <button
                                        key={preset}
                                        type="button"
                                        className={`tactical-btn epochs-preset ${getEpochs() === preset ? 'tactical-btn--primary' : ''}`}
                                        onClick={() => applyEpochs(preset)}
                                        disabled={training}
                                    >
                                        {preset}
                                    </button>
                                ))}
                            </div>
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
                        {modelType === 'cnn'
                            ? 'Enhanced CNN v2 • BatchNorm • Dropout • Early Stopping'
                            : 'EfficientNetB0 (ImageNet) • GAP • Dense(256) • Dropout'}
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
                        <div>
                            <div className="stat-value">{(trainResult.model_type || modelType).toUpperCase()}</div>
                            <div className="stat-label">Model Type</div>
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
