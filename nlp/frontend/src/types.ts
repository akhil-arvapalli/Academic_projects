export type ParserName = 'spacy' | 'nltk'

export interface TokenOut {
  id: number
  text: string
  lemma: string
  pos: string
  head_id: number
  dep_label: string
}

export interface ParseResponse {
  parser: ParserName
  tokens: TokenOut[]
  root_id: number
  note?: string
}
