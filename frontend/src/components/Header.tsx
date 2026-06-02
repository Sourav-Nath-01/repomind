'use client';

export function Header() {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-faint)',
      background: 'rgba(5,8,15,0.85)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <div style={{
        maxWidth: 1440,
        margin: '0 auto',
        padding: '0 28px',
        height: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
      }}>

        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Logo mark */}
          <div style={{
            width: 30, height: 30,
            borderRadius: 8,
            background: 'var(--grad-indigo)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(79,70,229,0.5)',
            flexShrink: 0,
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 4l4 4-4 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M9 12h4" stroke="white" strokeWidth="1.8" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div style={{
              fontWeight: 680,
              fontSize: 14,
              color: 'var(--text-100)',
              letterSpacing: '-0.02em',
              lineHeight: 1.2,
            }}>
              RepoMind
            </div>
            <div style={{
              fontSize: 10,
              color: 'var(--text-400)',
              letterSpacing: '0.04em',
              fontWeight: 500,
            }}>
              Autonomous Bug-Fix Agent
            </div>
          </div>
        </div>

        {/* Right side nav */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: '4px 10px',
            borderRadius: 99,
            background: 'rgba(16,185,129,0.08)',
            border: '1px solid rgba(16,185,129,0.18)',
          }}>
            <div style={{
              width: 6, height: 6,
              borderRadius: '50%',
              background: '#10B981',
              boxShadow: '0 0 6px #10B981',
              animation: 'pulse-ring 2.5s ease-in-out infinite',
            }} />
            <span style={{ fontSize: 11, color: '#34D399', fontWeight: 600, letterSpacing: '0.02em' }}>
              API Live
            </span>
          </div>

          <div style={{
            height: 18,
            width: 1,
            background: 'var(--border-dim)',
          }} />

          <a
            href="https://huggingface.co/spaces/SouravNath/repomind-api/tree/main"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '5px 11px',
              borderRadius: 6,
              background: 'var(--bg-hover)',
              border: '1px solid var(--border-dim)',
              color: 'var(--text-300)',
              fontSize: 11.5,
              fontWeight: 500,
              textDecoration: 'none',
              transition: 'all 0.15s',
              cursor: 'pointer',
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-base)';
              (e.currentTarget as HTMLElement).style.color = 'var(--text-100)';
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-dim)';
              (e.currentTarget as HTMLElement).style.color = 'var(--text-300)';
            }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            View Source
          </a>

          <span className="chip chip-dim" style={{ fontSize: 10.5 }}>
            SWE-bench Lite · GPT-4o-mini
          </span>
        </div>
      </div>
    </header>
  );
}
