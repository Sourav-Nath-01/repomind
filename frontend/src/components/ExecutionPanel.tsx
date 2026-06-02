'use client';

import { useEffect, useRef } from 'react';
import { useAgentStore } from '@/lib/store';

const STEPS = [
  { id: 1, label: 'Setup' },
  { id: 2, label: 'Clone' },
  { id: 3, label: 'Localise' },
  { id: 4, label: 'Patch' },
  { id: 5, label: 'Done' },
];

function StepIcon({ done, failed }: { done: boolean; failed?: boolean }) {
  if (failed) return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <path d="M2 2l8 8M10 2L2 10" stroke="white" strokeWidth="1.8" strokeLinecap="round"/>
    </svg>
  );
  if (done) return (
    <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
      <path d="M1.5 5.5l3 3 5-5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
  return null;
}

export function ExecutionPanel() {
  const { status, step, logs, localisedFiles, graphData, attempts } = useAgentStore();
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const isError = status === 'error';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* ── Pipeline progress ──────────────────────────────── */}
      <div className="panel" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <span className="section-label">
            <span className="pip" style={{ background: status === 'running' ? 'var(--indigo)' : status === 'done' ? 'var(--emerald)' : status === 'error' ? 'var(--rose)' : 'var(--text-500)' }} />
            Pipeline
          </span>
          <span className={`chip ${status === 'running' ? 'chip-indigo' : status === 'done' ? 'chip-emerald' : status === 'error' ? 'chip-rose' : 'chip-dim'}`} style={{ fontSize: 10.5 }}>
            {status === 'running' ? 'Running' : status === 'done' ? 'Complete' : status === 'error' ? 'Error' : 'Idle'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center' }}>
          {STEPS.map((s, i) => {
            const done    = s.id < step || status === 'done';
            const active  = s.id === step && status === 'running';
            const failed  = isError && s.id === step;

            return (
              <div key={s.id} style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
                  <div
                    className={`step-node ${done ? 'complete' : active ? 'active' : failed ? 'failed' : 'pending'}`}
                  >
                    <StepIcon done={done} failed={failed} />
                    {!done && !failed && (
                      <span style={{ fontSize: 10, fontWeight: 600 }}>{s.id}</span>
                    )}
                  </div>
                  <span style={{
                    fontSize: 9.5,
                    fontWeight: 500,
                    color: done || active ? 'var(--text-300)' : 'var(--text-500)',
                    letterSpacing: '0.02em',
                    whiteSpace: 'nowrap',
                    textTransform: 'uppercase',
                  }}>
                    {s.label}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`step-line ${done ? 'done' : ''}`} />
                )}
              </div>
            );
          })}
        </div>

        {/* Attempt chips */}
        {attempts.length > 0 && (
          <div style={{ display: 'flex', gap: 6, marginTop: 14, flexWrap: 'wrap' }}>
            {attempts.map(a => (
              <span key={a.attemptNum} className={`chip ${a.resolved ? 'chip-emerald' : 'chip-amber'}`} style={{ fontSize: 10.5 }}>
                Attempt {a.attemptNum} · {a.resolved ? 'Resolved' : a.failureCategory || 'Failed'}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* ── Files + Logs ───────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>

        {/* Localised files */}
        <div className="panel" style={{ padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span className="section-label">
              <span className="pip" style={{ background: 'var(--cyan)' }} />
              Localised Files
            </span>
            {localisedFiles.length > 0 && (
              <span className="chip chip-dim" style={{ fontSize: 10 }}>
                {localisedFiles.length} files
              </span>
            )}
          </div>

          {graphData && (
            <div style={{
              marginBottom: 10,
              padding: '7px 10px',
              background: 'var(--bg-surface)',
              borderRadius: 6,
              fontSize: 11,
              color: 'var(--text-400)',
              border: '1px solid var(--border-faint)',
              display: 'flex',
              gap: 12,
            }}>
              <span><span style={{ color: '#818CF8', fontWeight: 600 }}>{graphData.nodes.toLocaleString()}</span> nodes</span>
              <span><span style={{ color: '#67E8F9', fontWeight: 600 }}>{graphData.edges.toLocaleString()}</span> edges</span>
            </div>
          )}

          {localisedFiles.length === 0 ? (
            <div style={{
              color: 'var(--text-500)',
              fontSize: 12,
              padding: '24px 0',
              textAlign: 'center',
              fontStyle: 'italic',
            }}>
              {status === 'running' ? 'Localising relevant files…' : 'No files yet'}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {localisedFiles.map((fp, i) => (
                <div key={fp} className="file-row fade-up">
                  <span className="file-rank">{i + 1}</span>
                  <div style={{
                    width: 3, height: 14, borderRadius: 2, flexShrink: 0,
                    background: i === 0 ? 'var(--indigo)' : i === 1 ? 'var(--cyan)' : 'var(--border-dim)',
                  }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {fp}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Execution log */}
        <div className="panel" style={{ padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span className="section-label">
              <span className="pip" style={{
                background: status === 'running' ? 'var(--emerald)' : 'var(--text-500)',
                animation: status === 'running' ? 'pulse-ring 2s ease-in-out infinite' : 'none',
              }} />
              Execution Log
            </span>
            <span style={{ fontSize: 10.5, color: 'var(--text-500)', fontVariantNumeric: 'tabular-nums' }}>
              {logs.length} events
            </span>
          </div>

          <div style={{ height: 240, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
            {logs.length === 0 ? (
              <div style={{ color: 'var(--text-500)', fontSize: 12, padding: '24px 0', textAlign: 'center', fontStyle: 'italic' }}>
                Waiting for agent…
              </div>
            ) : (
              logs.map(log => (
                <div
                  key={log.id}
                  className={`log-entry ${log.type === 'success' ? 'is-success' : log.type === 'error' ? 'is-error' : log.type === 'warn' ? 'is-warn' : log.type === 'info' ? 'is-info' : ''}`}
                >
                  <span className="log-ts">
                    {new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  <span>{log.message}</span>
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
