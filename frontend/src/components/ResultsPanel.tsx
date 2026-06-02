'use client';

import { useAgentStore } from '@/lib/store';

function DiffLine({ line, index }: { line: string; index: number }) {
  const isAdd   = line.startsWith('+') && !line.startsWith('+++');
  const isDel   = line.startsWith('-') && !line.startsWith('---');
  const isMeta  = line.startsWith('@@');
  const isDim   = line.startsWith('diff') || line.startsWith('---') || line.startsWith('+++');

  return (
    <div className={`diff-line ${isAdd ? 'add' : isDel ? 'del' : isMeta ? 'meta' : isDim ? 'dim' : 'normal'}`}>
      <span className="diff-ln">{isDim || isMeta ? '' : index}</span>
      {line || ' '}
    </div>
  );
}

export function ResultsPanel() {
  const { resolved, currentPatch, attempts, totalTokens, elapsedSeconds, failureCategory } = useAgentStore();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* ── Result banner ───────────────────────────────── */}
      <div className={`result-banner ${resolved ? 'resolved' : 'failed'}`}>
        {/* Icon */}
        <div style={{
          width: 40, height: 40, borderRadius: '50%', flexShrink: 0,
          background: resolved ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)',
          border: `1px solid ${resolved ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          {resolved ? (
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M3 9l4.5 4.5 7.5-9" stroke="#10B981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 3l10 10M13 3L3 13" stroke="#F43F5E" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          )}
        </div>

        {/* Text */}
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: 16,
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: resolved ? '#34D399' : '#FB7185',
          }}>
            {resolved ? 'Issue Resolved' : 'Could Not Resolve'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-400)', marginTop: 2 }}>
            {attempts.length} attempt{attempts.length !== 1 ? 's' : ''}
            {' · '}{elapsedSeconds.toFixed(1)}s
            {' · '}{totalTokens.toLocaleString()} tokens
            {!resolved && failureCategory && ` · ${failureCategory}`}
          </div>
        </div>

        {/* Stat chips */}
        <div style={{ display: 'flex', gap: 16 }}>
          {[
            { label: 'Attempts', value: attempts.length },
            { label: 'Time',     value: `${elapsedSeconds.toFixed(1)}s` },
            { label: 'Tokens',   value: totalTokens.toLocaleString() },
          ].map(m => (
            <div key={m.label} style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-100)', letterSpacing: '-0.03em' }}>
                {m.value}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-400)', textTransform: 'uppercase', letterSpacing: '0.06em', marginTop: 1 }}>
                {m.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Attempt history ─────────────────────────────── */}
      {attempts.length > 0 && (
        <div className="panel" style={{ padding: '16px 18px' }}>
          <div className="section-label" style={{ marginBottom: 12 }}>
            <span className="pip" style={{ background: 'var(--amber)' }} />
            Attempt History
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {attempts.map(a => (
              <div key={a.attemptNum} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '10px 14px',
                background: 'var(--bg-surface)',
                borderRadius: 8,
                borderLeft: `2px solid ${a.resolved ? 'var(--emerald)' : 'var(--rose)'}`,
                border: `1px solid var(--border-faint)`,
                borderLeftWidth: 2,
              }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 6,
                  background: a.resolved ? 'var(--emerald-dim)' : 'var(--rose-dim)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {a.resolved ? (
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M1.5 6l3 3 6-6" stroke="#10B981" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  ) : (
                    <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                      <path d="M1 1l9 9M10 1L1 10" stroke="#F43F5E" strokeWidth="1.8" strokeLinecap="round"/>
                    </svg>
                  )}
                </div>

                <span style={{ fontWeight: 600, color: 'var(--text-200)', fontSize: 13 }}>
                  Attempt {a.attemptNum}
                </span>
                <span className={`chip ${a.resolved ? 'chip-emerald' : 'chip-rose'}`} style={{ fontSize: 10.5 }}>
                  {a.resolved ? 'Passed' : 'Failed'}
                </span>
                {!a.resolved && a.failureCategory && (
                  <span className="chip chip-amber" style={{ fontSize: 10.5 }}>
                    {a.failureCategory}
                  </span>
                )}
                {a.failToPassResults && (
                  <span style={{ fontSize: 11, color: 'var(--text-400)', marginLeft: 'auto' }}>
                    {Object.values(a.failToPassResults).filter(Boolean).length}
                    /{Object.keys(a.failToPassResults).length} tests passed
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Generated patch ──────────────────────────────── */}
      {currentPatch && (
        <div className="panel" style={{ padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span className="section-label">
              <span className="pip" style={{ background: 'var(--violet)' }} />
              Generated Patch
            </span>
            <button
              className="btn btn-ghost"
              style={{ padding: '5px 12px', fontSize: 11.5, gap: 5 }}
              onClick={() => navigator.clipboard.writeText(currentPatch)}
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <rect x="1" y="5" width="10" height="11" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M5 1h8a2 2 0 012 2v10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              Copy patch
            </button>
          </div>

          {/* Stats */}
          <div style={{
            display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap',
          }}>
            {(() => {
              const lines = currentPatch.split('\n');
              const added   = lines.filter(l => l.startsWith('+') && !l.startsWith('+++')).length;
              const removed = lines.filter(l => l.startsWith('-') && !l.startsWith('---')).length;
              const files   = lines.filter(l => l.startsWith('diff ')).length;
              return (
                <>
                  <span className="chip chip-emerald" style={{ fontSize: 10.5 }}>+{added} added</span>
                  <span className="chip chip-rose" style={{ fontSize: 10.5 }}>−{removed} removed</span>
                  <span className="chip chip-dim" style={{ fontSize: 10.5 }}>{files || 1} file{files > 1 ? 's' : ''}</span>
                </>
              );
            })()}
          </div>

          {/* Diff viewer */}
          <div className="diff-viewer">
            {currentPatch.split('\n').map((line, i) => (
              <DiffLine key={i} line={line} index={i + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
