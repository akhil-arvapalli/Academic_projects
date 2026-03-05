export function StatusBar({ status }) {
    return (
        <header className="status-bar">
            <div className="status-bar__left">
                <span className="status-bar__title">◈ SENTINEL</span>
                <span className="status-bar__subtitle">SATELLITE TERRAIN INTELLIGENCE</span>
            </div>
            <div className="status-bar__right">
                <div className="status-indicator">
                    <span className={`status-dot ${status.model_loaded ? '' : 'status-dot--inactive'}`}></span>
                    MODEL: {status.model_loaded ? 'ACTIVE' : 'OFFLINE'}
                </div>
                <div className="status-indicator">
                    <span className={`status-dot ${status.accuracy > 0 ? '' : 'status-dot--inactive'}`}></span>
                    ACC: {status.accuracy > 0 ? `${status.accuracy}%` : '—'}
                </div>
                <div className="status-indicator">
                    <span className={`status-dot ${status.features_loaded ? '' : 'status-dot--warning'}`}></span>
                    DATA: {status.features_loaded ? 'LOADED' : 'PENDING'}
                </div>
            </div>
        </header>
    )
}
