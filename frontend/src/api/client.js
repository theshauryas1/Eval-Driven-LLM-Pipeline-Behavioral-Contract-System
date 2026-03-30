const BASE = import.meta.env.VITE_API_URL || ''

async function req(path, opts = {}) {
    const res = await fetch(`${BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    })
    if (!res.ok) {
        const err = await res.text()
        throw new Error(err || res.statusText)
    }
    return res.json()
}

export const api = {
    getContracts: () => req('/contracts'),
    getResults: (params = {}) => {
        const q = new URLSearchParams(params).toString()
        return req(`/results${q ? '?' + q : ''}`)
    },
    getTraceDetail: (traceId) => req(`/results/${traceId}`),
    getStats: (contractId, days = 30) =>
        req(`/results/stats?contract_id=${contractId}&days=${days}`),
    postTrace: (payload) =>
        req('/trace', { method: 'POST', body: JSON.stringify(payload) }),
    health: () => req('/health'),
}
