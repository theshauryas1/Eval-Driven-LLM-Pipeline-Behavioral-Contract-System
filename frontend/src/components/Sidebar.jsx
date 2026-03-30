import { NavLink } from 'react-router-dom'
import { LayoutDashboard, ListChecks, TrendingDown, Zap } from 'lucide-react'

const links = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
    { to: '/traces', label: 'Traces', icon: ListChecks },
    { to: '/regression', label: 'Regression', icon: TrendingDown },
]

export default function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <div className="logo-dot">
                    <Zap size={14} color="#fff" />
                </div>
                <div>
                    <h1>ContractEval</h1>
                    <span>LLM Behavioral Contracts</span>
                </div>
            </div>

            <nav>
                {links.map(({ to, label, icon: Icon, end }) => (
                    <NavLink
                        key={to}
                        to={to}
                        end={end}
                        className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                    >
                        <Icon size={16} />
                        {label}
                    </NavLink>
                ))}
            </nav>

            <div style={{ marginTop: 'auto', padding: '1rem 0.5rem 0' }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4, color: 'var(--text-secondary)' }}>Stack</div>
                    FastAPI · Neon Postgres · LangGraph · Groq Llama 3.3 70B
                </div>
            </div>
        </aside>
    )
}
