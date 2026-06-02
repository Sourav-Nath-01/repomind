'use client';

import { Header }        from '@/components/Header';
import { MetricsBar }    from '@/components/MetricsBar';
import { SubmitPanel }   from '@/components/SubmitPanel';
import { ExecutionPanel} from '@/components/ExecutionPanel';
import { ResultsPanel }  from '@/components/ResultsPanel';
import { useAgentStore } from '@/lib/store';

export default function Home() {
  const { status } = useAgentStore();
  const hasResult = status === 'done' || status === 'error';

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>
      <Header />
      <MetricsBar />

      <main style={{
        maxWidth: 1440,
        margin: '0 auto',
        padding: '24px 28px 48px',
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '340px 1fr',
          gap: 16,
          alignItems: 'start',
        }}>

          {/* ── Left: Submit form ─────────────────────── */}
          <div className="panel" style={{ position: 'sticky', top: 72 }}>
            <SubmitPanel />
          </div>

          {/* ── Right: Execution + Results ────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <ExecutionPanel />
            {hasResult && <ResultsPanel />}
          </div>
        </div>
      </main>
    </div>
  );
}
