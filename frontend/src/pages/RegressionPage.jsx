import { useEffect, useState } from 'react'
import { api } from '../api/client'
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    CartesianGrid, ReferenceLine
} from 'recharts'
import { TrendingDown, AlertCircle } from 'lucide-react'

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
        <div style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '0.75rem 1rem',
            fontSize: 12,
        }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
            <div style={{ color: 'var(--accent-light)' }}>Pass rate: {d.pass_rate !== null ? `${d.pass_rate}%` : '-'}</div>
            <div style={{ color: 'var(--text-muted)' }}>Evals: {d.total} ({d.passed} passed)</div>
        </div>
    )
}

export default function RegressionPage() {
    const [contracts, setContracts] = useState([])
    const [selectedId, setSelectedId] = useState('')
    const [days, setDays] = useState(30)
    const [series, setSeries] = useState([])
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        api.getContracts().then(r => {
            setContracts(r.contracts)
            if (r.contracts.length) setSelectedId(r.contracts[0].id)
        }).catch(() => { })
    }, [])

    useEffect(() => {
        if (!selectedId) return
        setLoading(true)
        api.getStats(selectedId, days)
            .then(r => { setSeries(r.series); setLoading(false) })
            .catch(() => setLoading(false))
    }, [selectedId, days])

    const firstFail = series.find(d => d.pass_rate !== null && d.pass_rate < 90)
    const selected = contracts.find(c => c.id === selectedId)

    return (
        <div>
            <div className="page-header">
                <h2>Regression View</h2>
                <p>Detect when a contract started failing with historical pass rates and a first-failure marker.</p>
            </div>

            <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                <div>
                    <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>CONTRACT</label>
                    <select id="contract-select" value={selectedId} onChange={e => setSelectedId(e.target.value)}>
                        {contracts.map(c => (
                            <option key={c.id} value={c.id}>{c.id}</option>
                        ))}
                    </select>
                </div>
                <div>
                    <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>RANGE</label>
                    <select id="days-select" value={days} onChange={e => setDays(Number(e.target.value))}>
                        {[7, 14, 30, 60, 90].map(d => (
                            <option key={d} value={d}>Last {d} days</option>
                        ))}
                    </select>
                </div>

                {selected && (
                    <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                        <span className={`badge badge-${selected.type}`}>{selected.type}</span>
                        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4, maxWidth: 300 }}>
                            {selected.description}
                        </div>
                    </div>
                )}
            </div>

            {firstFail && (
                <div className="card" style={{
                    marginBottom: '1.5rem',
                    padding: '0.875rem 1.25rem',
                    border: '1px solid rgba(239,68,68,0.3)',
                    background: 'rgba(239,68,68,0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                }}>
                    <AlertCircle size={16} color="var(--red)" />
                    <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>
                        <strong style={{ color: 'var(--red)' }}>Regression detected</strong>{' '}
                        <strong>{selectedId}</strong> first failed on <strong>{firstFail.date}</strong> with a pass rate of{' '}
                        <strong style={{ color: 'var(--red)' }}>{firstFail.pass_rate}%</strong>.
                    </span>
                </div>
            )}

            <div className="card">
                <div className="card-header">
                    <span className="card-title">Pass Rate Over Time | {selectedId}</span>
                    <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-muted)' }}>
                        <span style={{ color: 'var(--green)' }}>Healthy (&ge;90%)</span>
                        <span style={{ color: 'var(--red)' }}>Failing (&lt;90%)</span>
                    </div>
                </div>

                {loading ? (
                    <div className="skeleton" style={{ height: 280 }} />
                ) : series.length === 0 ? (
                    <div className="empty-state" style={{ padding: '3rem 2rem' }}>
                        <TrendingDown size={32} style={{ margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
                        <h3>No data for this range</h3>
                        <p>Run some traces first to see the pass-rate history.</p>
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={series} margin={{ top: 8, right: 16, left: -8, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="date"
                                tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                                tickLine={false}
                            />
                            <YAxis
                                tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                                tickLine={false}
                                domain={[0, 100]}
                                tickFormatter={v => `${v}%`}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine
                                y={90}
                                stroke="rgba(239,68,68,0.4)"
                                strokeDasharray="6 4"
                                label={{ value: '90% threshold', fill: 'var(--red)', fontSize: 10, position: 'insideTopRight' }}
                            />
                            {firstFail && (
                                <ReferenceLine
                                    x={firstFail.date}
                                    stroke="rgba(239,68,68,0.6)"
                                    strokeDasharray="4 4"
                                    label={{ value: 'First fail', fill: 'var(--red)', fontSize: 10 }}
                                />
                            )}
                            <Line
                                type="monotone"
                                dataKey="pass_rate"
                                stroke="#7C3AED"
                                strokeWidth={2.5}
                                dot={{ r: 4, fill: '#7C3AED', strokeWidth: 0 }}
                                activeDot={{ r: 6, fill: '#a78bfa' }}
                                connectNulls
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    )
}
