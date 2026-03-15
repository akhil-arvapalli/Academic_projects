import type { ParseResponse, ParserName } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

export async function parseSentence(text: string, parser: ParserName): Promise<ParseResponse> {
  const response = await fetch(`${API_BASE}/parse`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, parser }),
  })

  if (!response.ok) {
    const fallback = `Parse failed with status ${response.status}`
    try {
      const details = await response.json()
      throw new Error(details.detail ?? fallback)
    } catch {
      throw new Error(fallback)
    }
  }

  return response.json() as Promise<ParseResponse>
}
