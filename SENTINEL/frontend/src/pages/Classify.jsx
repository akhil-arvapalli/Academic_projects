import { useState, useRef, useEffect } from 'react'
import { GlowCard, ConfidenceBar, TextReveal } from '../components/Components'
import { Upload, ScanSearch } from 'lucide-react'

const LABEL_ICONS = { 'Urban Land': '🏙️', 'Agricultural Land': '🌾', 'Range Land': '🏜️', 'Forest Land': '🌲' }

export default function Classify({ status, addLog }) {
    const [image, setImage] = useState(null)
    const [imageFile, setImageFile] = useState(null)
    const [result, setResult] = useState(null)
    const [overlayImage, setOverlayImage] = useState(null)
    const [loading, setLoading] = useState(false)
    const [dragActive, setDragActive] = useState(false)
    const [sampleImages, setSampleImages] = useState([])
    const fileRef = useRef()

    useEffect(() => {
        fetch('/api/sample-images')
            .then(r => r.json())
            .then(data => setSampleImages(data.images || []))
            .catch(() => { })
    }, [])

    const handleFile = (file) => {
        if (!file) return
        setImageFile(file)
        setResult(null)
        setOverlayImage(null)
        const reader = new FileReader()
        reader.onload = (e) => setImage(e.target.result)
        reader.readAsDataURL(file)
        addLog(`Image loaded: ${file.name}`)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragActive(false)
        handleFile(e.dataTransfer.files[0])
    }

    const handleSampleClick = async (name) => {
        addLog(`Loading sample: ${name}`)
        try {
            const res = await fetch(`/api/sample-images/${name}`)
            const blob = await res.blob()
            const file = new File([blob], name, { type: blob.type })
            handleFile(file)
        } catch {
            addLog('Failed to load sample image', 'error')
        }
    }

    const classify = async () => {
        if (!imageFile) return
        setLoading(true)
        setResult(null)
        addLog('Transmitting image for classification...')

        try {
            const form = new FormData()
            form.append('image', imageFile)
            const res = await fetch('/api/predict', { method: 'POST', body: form })
            const data = await res.json()

            if (data.success) {
                setResult(data.result)
                setOverlayImage(data.overlay_image)
                addLog(`Classification: ${data.result.label} (${data.result.confidence.toFixed(1)}%)`)
            } else {
                addLog(`Classification failed: ${data.error}`, 'error')
            }
        } catch {
            addLog('Server error during classification', 'error')
        }
        setLoading(false)
    }

    return (
        <div>
            <div className="page-header">
                <h1 className="page-header__title">◈ IMAGE CLASSIFICATION</h1>
                <p className="page-header__subtitle">Upload a satellite image for terrain analysis</p>
            </div>

            <div className="classify-layout">
                {/* Left: Upload + Preview */}
                <div>
                    {!image ? (
                        <div
                            className={`upload-zone ${dragActive ? 'upload-zone--active' : ''}`}
                            onClick={() => fileRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                            onDragLeave={() => setDragActive(false)}
                            onDrop={handleDrop}
                        >
                            <div className="upload-zone__icon">🛰️</div>
                            <div className="upload-zone__text">Drop satellite image here or click to upload</div>
                            <div className="upload-zone__hint">JPG, PNG, BMP supported</div>
                            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => handleFile(e.target.files[0])} />
                        </div>
                    ) : (
                        <div className="image-preview">
                            <img src={overlayImage || image} alt="Satellite" />
                            <div className="image-preview__overlay">
                                <div className="image-preview__scanline" />
                                <div className="image-preview__grid" />
                            </div>
                        </div>
                    )}

                    <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
                        {image && (
                            <>
                                <button className="tactical-btn tactical-btn--primary" onClick={classify} disabled={loading} style={{ flex: 1 }}>
                                    {loading ? <><span className="tactical-btn__spinner" />ANALYZING...</> : <><ScanSearch size={14} style={{ marginRight: 8, verticalAlign: 'middle' }} />CLASSIFY</>}
                                </button>
                                <button className="tactical-btn" onClick={() => { setImage(null); setResult(null); setOverlayImage(null); setImageFile(null) }}>
                                    CLEAR
                                </button>
                            </>
                        )}
                    </div>
                </div>

                {/* Right: Results */}
                <div className="classify-results">
                    {result ? (
                        <>
                            <GlowCard title="Classification Result">
                                <TextReveal text={result.label.toUpperCase()} />
                                <div className="text-dim text-mono mt-16" style={{ fontSize: 12 }}>
                                    Confidence: {result.confidence.toFixed(2)}%
                                </div>
                            </GlowCard>

                            <GlowCard title="Class Probabilities">
                                {Object.entries(result.probabilities).map(([label, prob]) => (
                                    <ConfidenceBar
                                        key={label}
                                        label={label}
                                        value={prob}
                                        icon={LABEL_ICONS[label]}
                                        isTop={label === result.label}
                                    />
                                ))}
                            </GlowCard>
                        </>
                    ) : (
                        <GlowCard title="Awaiting Input">
                            <div style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 13, lineHeight: 2 }}>
                                &gt; Upload an image to begin analysis<br />
                                &gt; Or select from samples below
                            </div>
                        </GlowCard>
                    )}

                    {/* Sample Images */}
                    {sampleImages.length > 0 && (
                        <GlowCard title="Sample Images">
                            <div className="sample-grid">
                                {sampleImages.map(name => (
                                    <div key={name} className="sample-thumb" onClick={() => handleSampleClick(name)}>
                                        <img src={`/api/sample-images/${name}`} alt={name} loading="lazy" />
                                    </div>
                                ))}
                            </div>
                        </GlowCard>
                    )}
                </div>
            </div>
        </div>
    )
}
