import { useEffect, useMemo, useState } from 'react'

import type { TokenOut } from '../types'

interface DependencyArcViewProps {
  tokens: TokenOut[]
  rootId: number
}

const DEP_COLORS: Record<string, string> = {
  ROOT: '#f25f5c',
  nsubj: '#247ba0',
  obj: '#70c1b3',
  amod: '#ff9f1c',
  det: '#5f0f40',
  advmod: '#3a86ff',
  prep: '#7b2cbf',
  punct: '#6c757d',
  dep: '#1982c4',
}

function colorForDep(dep: string): string {
  return DEP_COLORS[dep] ?? '#118ab2'
}

export function DependencyArcView({ tokens, rootId }: DependencyArcViewProps) {
  const [hoverText, setHoverText] = useState('Hover any token or arc to inspect details.')
  const [animationSeed, setAnimationSeed] = useState(0)

  useEffect(() => {
    if (tokens.length) {
      setAnimationSeed((prev) => prev + 1)
      setHoverText('Hover any token or arc to inspect details.')
    }
  }, [tokens, rootId])

  // ⚠️ hooks must be called unconditionally — BEFORE any early return
  const width = Math.max(920, tokens.length * 110)
  const baseY = 220

  const arcs = useMemo(
    () =>
      tokens
        .filter((token) => token.id !== token.head_id)
        .sort((a, b) => a.id - b.id),
    [tokens],
  )

  if (!tokens.length) {
    return (
      <div className="empty-graph">
        <span className="empty-icon">⬡</span>
        <span>Parse a sentence to generate the dependency graph.</span>
      </div>
    )
  }

  return (
    <>
      <div className="graph-scroll">
        <svg viewBox={`0 0 ${width} 300`} className="graph-svg" role="img" aria-label="Dependency graph">
          {arcs.map((token, arcIndex) => {
          const startX = 50 + token.head_id * 110
          const endX = 50 + token.id * 110
          const distance = Math.abs(endX - startX)
          const arcHeight = Math.max(45, distance * 0.45)
          const controlY = baseY - arcHeight
          const depColor = colorForDep(token.dep_label)

          const arrowX = endX
          const arrowY = baseY - 1

            return (
              <g
                key={`${animationSeed}-${token.id}-${token.head_id}`}
                className="arc-enter"
                style={{ animationDelay: `${arcIndex * 130}ms` }}
                onMouseEnter={() =>
                  setHoverText(`Arc ${token.head_id} -> ${token.id}: ${token.dep_label} (${token.text})`)
                }
              >
              <path
                d={`M ${startX} ${baseY} Q ${(startX + endX) / 2} ${controlY} ${endX} ${baseY}`}
                fill="none"
                stroke={depColor}
                strokeWidth="2.5"
              />
              <polygon
                points={`${arrowX - 5},${arrowY - 8} ${arrowX + 5},${arrowY - 8} ${arrowX},${arrowY + 1}`}
                fill={depColor}
              />
              <text x={(startX + endX) / 2} y={controlY - 8} textAnchor="middle" className="dep-label">
                {token.dep_label}
              </text>
              </g>
            )
          })}

          {tokens.map((token, tokenIndex) => {
            const x = 50 + token.id * 110
            const isRoot = token.id === rootId
            return (
              <g
                key={`${animationSeed}-${token.id}`}
                className="token-enter"
                style={{ animationDelay: `${tokenIndex * 110}ms` }}
                onMouseEnter={() =>
                  setHoverText(
                    `Token ${token.id}: ${token.text} | lemma=${token.lemma} | POS=${token.pos} | head=${token.head_id} | dep=${token.dep_label}`,
                  )
                }
              >
              <rect
                x={x - 40}
                y={236}
                width="80"
                height="38"
                rx="9"
                className={isRoot ? 'token-box root-token' : 'token-box'}
              />
              <text x={x} y={252} textAnchor="middle" className="token-text">
                {token.text}
              </text>
              <text x={x} y={267} textAnchor="middle" className="token-pos">
                {token.pos}
              </text>
              </g>
            )
          })}
        </svg>
      </div>
      <p className="graph-hover">{hoverText}</p>
    </>
  )
}
