'use client';

import { useEffect, useState } from 'react';
import type { Metrics } from '@/lib/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const DEMO: Partial<Metrics> = {
  total_issues_solved: 127,
  avg_elapsed_seconds: 48.3,
  avg_attempts: 1.8,
  recall_at_5: 0.74,
  avg_token_cost_per_issue: 3200,
};

export function MetricsBar() {
  const [metrics, setMetrics] = useState<Partial<Metrics>>(DEMO);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_URL}/api/metrics`);
        if (res.ok) setMetrics(await res.json());
      } catch { /* use demo */ }
    };
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  const items = [
    { label: 'Issues Resolved',   value: metrics.total_issues_solved ?? 0,                                   suffix: '',  accent: '#818CF8' },
    { label: 'Active Model',      value: 'GPT-4o-mini',                                                      suffix: '',  accent: '#67E8F9' },
    { label: 'Recall@5',          value: `${Math.round((metrics.recall_at_5 ?? 0) * 100)}`,                  suffix: '%', accent: '#94A3B8' },
    { label: 'Avg Attempts',      value: (metrics.avg_attempts ?? 0).toFixed(1),                             suffix: '',  accent: '#FCD34D' },
    { label: 'Avg Tokens',        value: Math.round(metrics.avg_token_cost_per_issue ?? 0).toLocaleString(), suffix: '',  accent: '#94A3B8' },
    { label: 'Avg Time',          value: Math.round(metrics.avg_elapsed_seconds ?? 0),                       suffix: 's', accent: '#94A3B8' },
  ];

  return (
    <div style={{
      background: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border-faint)',
    }}>
      <div style={{
        maxWidth: 1440,
        margin: '0 auto',
        padding: '0 28px',
        display: 'flex',
        overflowX: 'auto',
      }}>
        {items.map((item, i) => (
          <div
            key={item.label}
            style={{
              flex: '1 0 auto',
              padding: '12px 24px',
              borderRight: i < items.length - 1 ? '1px solid var(--border-faint)' : 'none',
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
            }}
          >
            <div style={{
              fontSize: 19,
              fontWeight: 700,
              color: item.accent,
              letterSpacing: '-0.03em',
              fontVariantNumeric: 'tabular-nums',
              lineHeight: 1,
            }}>
              {item.value}{item.suffix}
            </div>
            <div style={{
              fontSize: 10,
              fontWeight: 500,
              color: 'var(--text-400)',
              textTransform: 'uppercase',
              letterSpacing: '0.07em',
              marginTop: 3,
            }}>
              {item.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
