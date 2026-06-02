'use client';

import { useAgentStore } from '@/lib/store';

const EXAMPLE_ISSUE = `Django: QuerySet.filter() raises TypeError when combining Q objects with annotated fields.

Steps to reproduce:
  qs = MyModel.objects.annotate(total=Sum('items__price'))
  qs.filter(Q(total__gt=100) | Q(name='test'))

Expected: queryset returned
Actual: TypeError: unsupported operand type(s) for |: 'Q' and 'Q'`;

const LabelRow = ({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
    <label className="field-label" style={{ margin: 0 }}>{children}</label>
    {right}
  </div>
);

export function SubmitPanel() {
  const {
    repo, problemStatement, maxAttempts, topKFiles,
    status, setField, submitJob, reset,
  } = useAgentStore();

  const isRunning = status === 'running' || status === 'queued';
  const isDone    = status === 'done' || status === 'error';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>

      {/* Panel header */}
      <div style={{
        padding: '20px 22px 16px',
        borderBottom: '1px solid var(--border-faint)',
      }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-100)', letterSpacing: '-0.02em' }}>
          Submit Issue
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-400)', marginTop: 3 }}>
          Paste a GitHub issue and the agent will localise, patch, and verify.
        </div>
      </div>

      {/* Form body */}
      <div style={{ padding: '20px 22px', display: 'flex', flexDirection: 'column', gap: 16 }}>

        {/* Repository */}
        <div>
          <label className="field-label">Repository</label>
          <div style={{ position: 'relative' }}>
            <div style={{
              position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-500)', display: 'flex', alignItems: 'center',
            }}>
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" opacity="0.8">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
              </svg>
            </div>
            <input
              className="input"
              style={{ paddingLeft: 32 }}
              placeholder="owner/repo  e.g. django/django"
              value={repo}
              onChange={e => setField('repo', e.target.value)}
              disabled={isRunning}
            />
          </div>
        </div>

        {/* Problem statement */}
        <div>
          <LabelRow right={
            <button
              className="btn btn-ghost"
              style={{ padding: '3px 9px', fontSize: 11 }}
              onClick={() => {
                setField('repo', 'django/django');
                setField('problemStatement', EXAMPLE_ISSUE);
              }}
              disabled={isRunning}
            >
              Load example
            </button>
          }>
            Problem Statement
          </LabelRow>
          <textarea
            className="textarea"
            placeholder="Paste the GitHub issue description — include error messages and steps to reproduce for best results."
            value={problemStatement}
            onChange={e => setField('problemStatement', e.target.value)}
            disabled={isRunning}
            rows={7}
          />
        </div>

        {/* Config */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            { key: 'maxAttempts', label: 'Max Attempts', min: 1, max: 5, val: maxAttempts },
            { key: 'topKFiles',   label: 'Top-K Files',  min: 1, max: 20, val: topKFiles },
          ].map(f => (
            <div key={f.key}>
              <label className="field-label">{f.label}</label>
              <input
                className="input"
                type="number"
                min={f.min} max={f.max}
                value={f.val}
                onChange={e => setField(f.key as 'maxAttempts' | 'topKFiles', parseInt(e.target.value))}
                disabled={isRunning}
              />
            </div>
          ))}
        </div>

        {/* Action */}
        {!isDone ? (
          <button className="btn-run" onClick={submitJob} disabled={isRunning}>
            {isRunning ? (
              <><span className="spinner" /> Running agent...</>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <path d="M3 3l10 5-10 5V3z" fill="white" stroke="white" strokeWidth="0.5" strokeLinejoin="round"/>
                </svg>
                Run Agent
              </>
            )}
          </button>
        ) : (
          <button className="btn-reset" onClick={reset}>
            ← New Issue
          </button>
        )}
      </div>

      {/* Pipeline info footer */}
      <div style={{
        padding: '14px 22px',
        borderTop: '1px solid var(--border-faint)',
        fontSize: 11,
        color: 'var(--text-400)',
        lineHeight: 1.8,
      }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 6px', alignItems: 'center' }}>
          {['BM25', 'PPR Graph', 'RRF Fusion', 'Llama-3.3-70B', 'git apply', 'pytest'].map((step, i, arr) => (
            <span key={step} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{
                background: 'var(--bg-overlay)',
                border: '1px solid var(--border-faint)',
                borderRadius: 4,
                padding: '1px 7px',
                fontSize: 10.5,
                fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--text-300)',
              }}>
                {step}
              </span>
              {i < arr.length - 1 && (
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <path d="M1 4h6M4 1l3 3-3 3" stroke="var(--text-500)" strokeWidth="1" strokeLinecap="round"/>
                </svg>
              )}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
