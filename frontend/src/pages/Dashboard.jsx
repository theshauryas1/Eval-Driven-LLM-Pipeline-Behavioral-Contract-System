import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { LineChart, Line, ResponsiveContainer } from 'recharts'
import { CheckCircle, PlayCircle } from 'lucide-react'

function PassRateSparkline({ contractId }) {
    const [data, setData] = useState([])

    useEffect(() => {
        api.getStats(contractId, 14).then(r => setData(r.series)).catch(() => { })
    }, [contractId])

    if (!data.length) {
        return <div style={{ height: 48, opacity: 0.3, fontSize: 12, color: 'var(--text-muted)' }}>No data yet</div>
    }

    return (
        <ResponsiveContainer width="100%" height={48}>
            <LineChart data={data}>
                <Line
                    type="monotone"
                    dataKey="pass_rate"
                    stroke="#7C3AED"
                    strokeWidth={2}
                    dot={false}
                />
            </LineChart>
        </ResponsiveContainer>
    )
}

function ContractCard({ contract }) {
    const rate = contract.pass_rate
    const color = rate === null ? '#5a5a7a' : rate >= 90 ? '#22c55e' : rate >= 70 ? '#f59e0b' : '#ef4444'

    return (
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card-header">
                <span className="card-title" style={{ fontSize: 13 }}>{contract.id.replace(/_/g, ' ')}</span>
                <span className={`badge badge-${contract.type}`}>{contract.type}</span>
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                {contract.description}
            </p>
            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Pass rate</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color }}>
                        {rate !== null ? `${rate}%` : '-'}
                    </span>
                </div>
                <div className="pass-bar-wrap">
                    <div className="pass-bar-fill" style={{ width: `${rate ?? 0}%`, background: color }} />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    {contract.eval_count} evaluation{contract.eval_count !== 1 ? 's' : ''}
                </div>
            </div>
            <PassRateSparkline contractId={contract.id} />
        </div>
    )
}

function StatTile({ value, label, color }) {
    return (
        <div className="card stat-tile">
            <div className="stat-value" style={{ color }}>{value}</div>
            <div className="stat-label">{label}</div>
        </div>
    )
}

export default function Dashboard() {
    const [contracts, setContracts] = useState([])
    const [loading, setLoading] = useState(true)
    const [backendOk, setBackendOk] = useState(null)
    const [demoRunning, setDemoRunning] = useState(false)
    const [demoMessage, setDemoMessage] = useState('')

    async function loadContracts() {
        setLoading(true)
        try {
            const response = await api.getContracts()
            setContracts(response.contracts)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        api.health()
            .then(() => setBackendOk(true))
            .catch(() => setBackendOk(false))
        loadContracts().catch(() => { })
    }, [])

    async function handleRunDemo() {
        setDemoRunning(true)
        setDemoMessage('')
        try {
            const result = await api.runDemo('hallucination')
            setDemoMessage(`Demo trace created with ${result.summary.failed} failing contract(s). Open Traces to inspect it.`)
            await loadContracts()
        } catch {
            setDemoMessage('Demo run failed. Check that the backend is online and try again.')
        } finally {
            setDemoRunning(false)
        }
    }

    const total = contracts.length
    const failing = contracts.filter(c => c.pass_rate !== null && c.pass_rate < 90).length
    const avgRate = total
        ? Math.round(contracts.filter(c => c.pass_rate !== null).reduce((s, c) => s + c.pass_rate, 0) / (contracts.filter(c => c.pass_rate !== null).length || 1))
        : null

    return (
        <div>
            <div className="page-header">
                <h2>Contract Dashboard</h2>
                <p>
                    Monitor behavioral contracts across your LLM pipeline.&nbsp;
                    {backendOk === false && (
                        <span style={{ color: 'var(--red)' }}>Backend offline. Start the FastAPI server.</span>
                    )}
                    {backendOk === true && (
                        <span style={{ color: 'var(--green)' }}>Backend connected</span>
                    )}
                </p>
            </div>

            <div className="card" style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
                <div>
                    <div className="card-title" style={{ marginBottom: 4 }}>Instant Demo</div>
                    <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        Seed a preloaded hallucination example and inspect the failed contracts in under a minute.
                    </p>
                </div>
                <button className="btn btn-primary" onClick={handleRunDemo} disabled={demoRunning || backendOk === false}>
                    <PlayCircle size={15} />
                    {demoRunning ? 'Running demo...' : 'Run Demo Trace'}
                </button>
                {demoMessage && (
                    <div style={{ width: '100%', fontSize: 12, color: 'var(--accent-light)' }}>
                        {demoMessage}
                    </div>
                )}
            </div>

            <div className="grid-3" style={{ marginBottom: '2rem' }}>
                <StatTile value={total} label="Active Contracts" color="var(--accent-light)" />
                <StatTile
                    value={avgRate !== null ? `${avgRate}%` : '-'}
                    label="Avg Pass Rate"
                    color={avgRate >= 90 ? 'var(--green)' : avgRate >= 70 ? 'var(--yellow)' : 'var(--red)'}
                />
                <StatTile value={failing} label="Contracts Failing" color={failing > 0 ? 'var(--red)' : 'var(--green)'} />
            </div>

            {loading ? (
                <div className="grid-auto">
                    {[1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="card skeleton" style={{ height: 180 }} />
                    ))}
                </div>
            ) : contracts.length === 0 ? (
                <div className="empty-state">
                    <CheckCircle size={40} style={{ margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
                    <h3>No contracts yet</h3>
                    <p>Start the backend. Contracts sync from YAML on startup.</p>
                </div>
            ) : (
                <div className="grid-auto">
                    {contracts.map(c => <ContractCard key={c.id} contract={c} />)}
                </div>
            )}
        </div>
    )
}
