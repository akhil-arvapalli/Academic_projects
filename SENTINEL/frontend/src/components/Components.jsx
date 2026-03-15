export function ConfidenceBar({ label, value, icon, isTop }) {
    return (
        <div className={`confidence-bar ${isTop ? 'confidence-bar--top' : value < 10 ? 'confidence-bar--amber' : ''}`}>
            <div className="confidence-bar__header">
                <span className="confidence-bar__label">
                    {icon && <span>{icon}</span>}
                    {label}
                </span>
                <span className="confidence-bar__value">{value.toFixed(1)}%</span>
            </div>
            <div className="confidence-bar__track">
                <div className="confidence-bar__fill" style={{ width: `${value}%` }} />
            </div>
        </div>
    )
}

export function LogConsole({ logs }) {
    return (
        <div className="log-console">
            {logs.length === 0 && (
                <div className="log-console__line" style={{ color: 'var(--text-dim)' }}>
                    Awaiting system events...
                </div>
            )}
            {logs.map((log, i) => (
                <div key={i} className="log-console__line" style={{ animationDelay: `${i * 0.05}s` }}>
                    <span className="log-console__timestamp">[{log.time}]</span>
                    <span className={`log-console__${log.level}`}>{log.message}</span>
                </div>
            ))}
        </div>
    )
}

export function GlowCard({ title, children, className = '' }) {
    return (
        <div className={`glow-card ${className}`}>
            {title && <div className="glow-card__title">{title}</div>}
            {children}
        </div>
    )
}

export function TextReveal({ text, className = '' }) {
    return (
        <div className={`result-label ${className}`}>
            {text.split('').map((char, i) => (
                <span key={i} style={{ animationDelay: `${i * 0.04}s` }}>
                    {char === ' ' ? '\u00A0' : char}
                </span>
            ))}
        </div>
    )
}
