import { useState } from 'react'
import { parseSentence } from './api'
import { DependencyArcView } from './components/DependencyArcView'
import { useParser } from './hooks/useParser'
import type { ParseResponse } from './types'
import './App.css'

// POS tag → CSS class (colour comes from App.css .pos-XXX rules)
const POS_SET = new Set([
  'NOUN','PROPN','VERB','AUX','ADJ','ADV','DET','NUM',
  'PUNCT','ADP','PART','PRON','SCONJ','CCONJ','CONJ',
])
function posClass(pos: string) {
  return POS_SET.has(pos) ? `pos-badge pos-${pos}` : 'pos-badge'
}

function depClass(dep: string) {
  const known = ['ROOT','nsubj','obj','dobj','amod','det','advmod','prep','pobj']
  return known.includes(dep) ? `dep-badge dep-${dep}` : 'dep-badge'
}

const EXAMPLES = [
  'The quick brown fox jumps over the lazy dog.',
  'I saw the man with a telescope near the river.',
  'Students who practice consistently improve faster than expected.',
]

function App() {
  const { text, setText, parser, setParser, data, loading, error, tokenSummary, runParse } =
    useParser(EXAMPLES[0], 'spacy')
  const [compareMode, setCompareMode] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState('')
  const [compareSpacy, setCompareSpacy] = useState<ParseResponse | null>(null)
  const [compareNltk, setCompareNltk] = useState<ParseResponse | null>(null)

  async function handlePrimaryParse() {
    if (compareMode) {
      setCompareLoading(true)
      setCompareError('')
      try {
        const [spacyResult, nltkResult] = await Promise.all([
          parseSentence(text, 'spacy'),
          parseSentence(text, 'nltk'),
        ])
        setCompareSpacy(spacyResult)
        setCompareNltk(nltkResult)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unexpected error while comparing parsers.'
        setCompareError(message)
        setCompareSpacy(null)
        setCompareNltk(null)
      } finally {
        setCompareLoading(false)
      }
      return
    }

    await runParse()
  }

  const activeLoading = compareMode ? compareLoading : loading
  const activeError = compareMode ? compareError : error

  return (
    <div className="app-shell">
      {/* ––– Site header ––– */}
      <header className="site-header">
        <div className="logo">
          <span className="logo-mark" aria-hidden="true">⧛</span>
          <span>DepViz<span className="logo-sub"> &nbsp;Dependency Visualizer</span></span>
        </div>
        <span className="header-badge">MVP • v0.1</span>
      </header>

      {/* ––– Hero ––– */}
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">● NLP Studio</span>
          <h1>Parse &amp; Visualize<br /><span>Sentence Structure</span></h1>
          <p>
            Inspect head–dependent arcs in real time using spaCy or NLTK.
            Toggle parsers, compare them side-by-side, and explore token-level detail.
          </p>
          <div className="hero-pills">
            <span className="pill spacy">spaCy</span>
            <span className="pill nltk">NLTK</span>
            <span className="pill dark">Real-time</span>
          </div>
        </div>
        <div className="hero-glow" aria-hidden="true">
          <div className="orb orb-a"></div>
          <div className="orb orb-b"></div>
          <div className="orb orb-c"></div>
          <div className="orb orb-d"></div>
        </div>
      </section>

      <section className="control-panel">
        <label htmlFor="sentence" className="input-label">
          Sentence
        </label>
        <textarea
          id="sentence"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={3}
          placeholder="Type a sentence to parse..."
        />

        <div className="control-row">
          <div className="mode-toggle" role="group" aria-label="View mode selection">
            <button
              className={!compareMode ? 'active' : ''}
              onClick={() => setCompareMode(false)}
              type="button"
            >
              Single
            </button>
            <button
              className={compareMode ? 'active' : ''}
              onClick={() => setCompareMode(true)}
              type="button"
            >
              Compare
            </button>
          </div>

          <div className="parser-toggle" role="group" aria-label="Parser selection">
            <button
              className={parser === 'spacy' ? 'active' : ''}
              onClick={() => setParser('spacy')}
              type="button"
              disabled={compareMode}
            >
              spaCy
            </button>
            <button
              className={parser === 'nltk' ? 'active' : ''}
              onClick={() => setParser('nltk')}
              type="button"
              disabled={compareMode}
            >
              NLTK
            </button>
          </div>
          <button onClick={handlePrimaryParse} type="button" className="parse-btn" disabled={activeLoading || !text.trim()}>
            {activeLoading ? (
              <><span className="spinner" />Parsing&hellip;</>
            ) : compareMode ? 'Compare Parsers' : 'Parse Sentence'}
          </button>
        </div>

        <div className="example-row">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => setText(example)} className="example-chip">
              {example}
            </button>
          ))}
        </div>

        <div className="status-bar">
          <span className={`status-dot ${activeLoading ? 'loading' : activeError ? 'error' : (data || compareSpacy) ? 'ready' : ''}`} />
          <span>{compareMode ? 'Compare mode — both parsers run simultaneously.' : (tokenSummary || 'Enter a sentence above to begin.')}</span>
        </div>
        {!compareMode && data?.note ? <p className="note-msg">⚠️ {data.note}</p> : null}
        {compareMode && compareNltk?.note ? <p className="note-msg">⚠️ {compareNltk.note}</p> : null}
        {activeError ? <p className="error-msg"><span>✕</span> {activeError}</p> : null}
      </section>

      {!compareMode ? (
        <section className="graph-panel">
          <div className="panel-header">
            <h2>Dependency Graph</h2>
            <span className="panel-tag">{parser === 'spacy' ? 'spaCy' : 'NLTK'} • animated</span>
          </div>
          <DependencyArcView tokens={data?.tokens ?? []} rootId={data?.root_id ?? 0} />
        </section>
      ) : (
        <div className="compare-grid">
          <section className="graph-panel">
            <div className="panel-header">
              <h2 className="spacy-label">spaCy</h2>
              <span className="panel-tag">true dep parse</span>
            </div>
            <DependencyArcView tokens={compareSpacy?.tokens ?? []} rootId={compareSpacy?.root_id ?? 0} />
          </section>
          <section className="graph-panel">
            <div className="panel-header">
              <h2 className="nltk-label">NLTK</h2>
              <span className="panel-tag">heuristic</span>
            </div>
            <DependencyArcView tokens={compareNltk?.tokens ?? []} rootId={compareNltk?.root_id ?? 0} />
          </section>
        </div>
      )}

      <section className="tokens-panel">
        <div className="panel-header">
          <h2>Token Details</h2>
          {compareMode && <span className="panel-tag">spaCy</span>}
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Token</th>
                <th>Lemma</th>
                <th>POS</th>
                <th>Head</th>
                <th>Dependency</th>
              </tr>
            </thead>
            <tbody>
              {((compareMode ? compareSpacy : data)?.tokens ?? []).map((token) => (
                <tr key={token.id}>
                  <td className="td-id">{token.id}</td>
                  <td className="td-token">{token.text}</td>
                  <td className="td-lemma">{token.lemma}</td>
                  <td><span className={posClass(token.pos)}>{token.pos}</span></td>
                  <td className="td-id">{token.head_id}</td>
                  <td><span className={depClass(token.dep_label)}>{token.dep_label}</span></td>
                </tr>
              ))}
              {((compareMode ? compareSpacy : data)?.tokens ?? []).length === 0 && (
                <tr><td colSpan={6} style={{ textAlign:'center', padding:'24px', color:'var(--text-muted)', fontSize:'0.83rem' }}>No tokens yet — run a parse above.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="site-footer">
        DepViz &mdash; Dependency Parser Visualization &nbsp;··&nbsp; spaCy &amp; NLTK &nbsp;··&nbsp; Built with FastAPI + React
      </footer>
    </div>
  )
}

export default App
