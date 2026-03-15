import { useMemo, useState } from 'react'

import { parseSentence } from '../api'
import type { ParseResponse, ParserName } from '../types'

export function useParser(initialText: string, initialParser: ParserName = 'spacy') {
  const [text, setText] = useState(initialText)
  const [parser, setParser] = useState<ParserName>(initialParser)
  const [data, setData] = useState<ParseResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const tokenSummary = useMemo(() => {
    if (!data?.tokens.length) {
      return 'No parsed tokens yet.'
    }
    return `${data.tokens.length} token(s), parser: ${data.parser.toUpperCase()}, root index: ${data.root_id}`
  }, [data])

  async function runParse() {
    setLoading(true)
    setError('')
    try {
      const response = await parseSentence(text, parser)
      setData(response)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error while parsing.'
      setError(message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  return {
    text,
    setText,
    parser,
    setParser,
    data,
    loading,
    error,
    tokenSummary,
    runParse,
  }
}
