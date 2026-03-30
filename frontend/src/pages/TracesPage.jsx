import { useEffect, useState, useCallback } from 'react'
import { api } from '../api/client'
import { X, CheckCircle, XCircle, ChevronRight } from 'lucide-react'

function ReasoningTrace({ steps }) {
    if (!steps || steps.length === 0) return null
    return (
        <div style={{ marginTop: '1rem' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-light)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
                LangGraph Reasoning Trace
            </div>
            {steps.map((step, i) => (
                <div key={i} className="trace-step">
                    <div className="trace-step-title">Step {i + 1} — {step.step}</div>
                    <div className="code-block">
                        {typeof step.result === 'string'
                            ? step.result
                            : JSON.stringify(step.result, null, 2)}
                    </div>
                </div>
            ))}
        </div>
    )
}

function TraceInspectorModal({ traceId, onClose }) {
    const [detail, setDetail] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        api.getTraceDetail(traceId)
            .then(r => { setDetail(r); setLoading(false) })
            .catch(() => setLoading(false))
    }, [traceId])

    return (
        <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
            <div className="modal">
                <div className="modal-header">
                    <div>
                        <div className="card-title">Trace Inspector</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>
                            {traceId}
                        </div>
                    </div>
                    <button className="modal-close" onClick={onClose}><X size={18} /></button>
                </div>
                <div className="modal-body">
                    {loading ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            {[1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 60 }} />)}
                        </div>
                    ) : !detail ? (
                        <div className="empty-state"><p>Failed to load trace detail.</p></div>
                    ) : (
                        <div>
                            {/* Summary */}
                            <div className="grid-3" style={{ marginBottom: '1.25rem' }}>
                                <div className="card" style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--green)' }}>{detail.summary.passed}</div>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>PASSED</div>
                                </div>
                                <div className="card" style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--red)' }}>{detail.summary.failed}</div>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>FAILED</div>
                                </div>
                                <div className="card" style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--accent-light)' }}>{detail.summary.total_contracts}</div>
                                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>TOTAL</div>
                                </div>
                            </div>

                            {/* Input / Context / Output */}
                            {[
                                { label: 'Input', value: detail.trace.input_text },
                                { label: 'Retrieved Context', value: detail.trace.retrieved_context },
                                { label: 'Output', value: detail.trace.output },
                            ].map(({ label, value }) => value ? (
                                <div key={label} style={{ marginBottom: '1rem' }}>
                                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>{label}</div>
                                    <div className="code-block" style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{value}</div>
                                </div>
                            ) : null)}

                            {/* Per-contract results */}
                            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8, marginTop: 8 }}>
                                Contract Results
                            </div>
                            {detail.eval_results.map(r => (
                                <div key={r.id} style={{ marginBottom: 12 }}>
                                    <div className="card" style={{ padding: '0.875rem 1rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                            {r.passed
                                                ? <CheckCircle size={16} color="var(--green)" />
                                                : <XCircle size={16} color="var(--red)" />}
                                            <span style={{ fontWeight: 600, fontSize: 13 }}>{r.contract_id}</span>
                                            <span className={`badge badge-${r.passed ? 'pass' : 'fail'}`} style={{ marginLeft: 'auto' }}>
                                                {r.passed ? 'PASS' : 'FAIL'}
                                            </span>
                                        </div>
                                        <div style={{ marginTop: 8, fontSize: 12.5, color: 'var(--text-secondary)', paddingLeft: 26 }}>
                                            {r.explanation}
                                        </div>
                                        {r.reasoning_trace && r.reasoning_trace.length > 0 && (
                                            <ReasoningTrace steps={r.reasoning_trace} />
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default function TracesPage() {
    const [traces, setTraces] = useState([])
    const [total, setTotal] = useState(0)
    const [loading, setLoading] = useState(true)
    const [selectedTrace, setSelectedTrace] = useState(null)
    const [pipelineFilter, setPipelineFilter] = useState('')
    const [page, setPage] = useState(0)
    const LIMIT = 20

    const load = useCallback(() => {
        setLoading(true)
        const params = { limit: LIMIT, offset: page * LIMIT }
        if (pipelineFilter) params.pipeline_id = pipelineFilter
        api.getResults(params)
            .then(r => { setTraces(r.items); setTotal(r.total); setLoading(false) })
            .catch(() => setLoading(false))
    }, [pipelineFilter, page])

    useEffect(() => { load() }, [load])

    const fmtDate = (iso) => iso
        ? new Date(iso).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' })
        : '—'

    return (
        <div>
            <div className="page-header">
                <h2>Trace Inspector</h2>
                <p>Every LLM pipeline call — click any row to see per-contract verdicts and reasoning traces.</p>
            </div>

            <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: 12 }}>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>Pipeline ID</label>
                <input
                    id="pipeline-filter"
                    placeholder="Filter by pipeline_id…"
                    value={pipelineFilter}
                    onChange={e => { setPipelineFilter(e.target.value); setPage(0) }}
                    style={{ width: 240 }}
                />
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                    {total} trace{total !== 1 ? 's' : ''}
                </span>
            </div>

            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <div className="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Pipeline</th>
                                <th>Input (truncated)</th>
                                <th>Pass</th>
                                <th>Fail</th>
                                <th>Violations</th>
                                <th>When</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {loading ? (
                                Array.from({ length: 8 }).map((_, i) => (
                                    <tr key={i}>
                                        {Array.from({ length: 7 }).map((_, j) => (
                                            <td key={j}><div className="skeleton" style={{ height: 14, width: '80%' }} /></td>
                                        ))}
                                    </tr>
                                ))
                            ) : traces.length === 0 ? (
                                <tr>
                                    <td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '3rem' }}>
                                        No traces yet. POST to /trace to get started.
                                    </td>
                                </tr>
                            ) : traces.map(t => (
                                <tr key={t.id} onClick={() => setSelectedTrace(t.id)}>
                                    <td style={{ fontWeight: 600, fontSize: 12 }}>{t.pipeline_id}</td>
                                    <td style={{ color: 'var(--text-secondary)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {t.input_text}
                                    </td>
                                    <td><span style={{ color: 'var(--green)', fontWeight: 700 }}>{t.eval_summary.passed}</span></td>
                                    <td><span style={{ color: t.eval_summary.failed > 0 ? 'var(--red)' : 'var(--text-muted)', fontWeight: 700 }}>{t.eval_summary.failed}</span></td>
                                    <td style={{ maxWidth: 180 }}>
                                        {t.eval_summary.violations.length > 0 ? (
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                                {t.eval_summary.violations.slice(0, 2).map(v => (
                                                    <span key={v} className="badge badge-fail" style={{ fontSize: 10 }}>{v.replace(/_/g, ' ')}</span>
                                                ))}
                                                {t.eval_summary.violations.length > 2 && (
                                                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>+{t.eval_summary.violations.length - 2}</span>
                                                )}
                                            </div>
                                        ) : (
                                            <span style={{ color: 'var(--green)', fontSize: 12 }}>None</span>
                                        )}
                                    </td>
                                    <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{fmtDate(t.created_at)}</td>
                                    <td><ChevronRight size={14} color="var(--text-muted)" /></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                {total > LIMIT && (
                    <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid var(--border)', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                        <button className="btn btn-ghost" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>
                            ← Prev
                        </button>
                        <span style={{ fontSize: 13, color: 'var(--text-muted)', padding: '0.5rem 0.5rem' }}>
                            {page + 1} / {Math.ceil(total / LIMIT)}
                        </span>
                        <button className="btn btn-ghost" onClick={() => setPage(p => p + 1)} disabled={(page + 1) * LIMIT >= total}>
                            Next →
                        </button>
                    </div>
                )}
            </div>

            {selectedTrace && (
                <TraceInspectorModal
                    traceId={selectedTrace}
                    onClose={() => setSelectedTrace(null)}
                />
            )}
        </div>
    )
}
