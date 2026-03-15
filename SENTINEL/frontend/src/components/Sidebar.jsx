import { LayoutDashboard, ScanSearch, Brain, BarChart3 } from 'lucide-react'

const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'classify', label: 'Classify', icon: ScanSearch },
    { id: 'training', label: 'Training', icon: Brain },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
]

export function Sidebar({ activePage, onNavigate }) {
    return (
        <aside className="sidebar">
            <nav className="sidebar__nav">
                {navItems.map(item => (
                    <button
                        key={item.id}
                        className={`sidebar__link ${activePage === item.id ? 'sidebar__link--active' : ''}`}
                        onClick={() => onNavigate(item.id)}
                    >
                        <item.icon className="sidebar__icon" size={18} />
                        {item.label}
                    </button>
                ))}
            </nav>
            <div className="sidebar__footer">
                SENTINEL v2.0.0<br />
                TensorFlow 2.20
            </div>
        </aside>
    )
}
