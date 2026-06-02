// src/lib/store.ts
// ──────────────────
// Zustand global state store — single source of truth for the agent UI

import { create } from 'zustand';
import type { AttemptRecord, GraphData, LogEntry, TaskStatus } from './types';

interface AgentStore {
  // Form state
  repo: string;
  problemStatement: string;
  maxAttempts: number;
  topKFiles: number;
  setField: (key: string, value: unknown) => void;

  // Execution state
  taskId: string;
  status: TaskStatus;
  step: number;
  totalSteps: number;

  // Results
  localisedFiles: string[];
  attempts: AttemptRecord[];
  currentPatch: string;
  resolved: boolean;
  failureCategory: string;
  totalTokens: number;
  elapsedSeconds: number;
  graphData: GraphData | null;

  // Logs
  logs: LogEntry[];

  // Actions
  submitJob: () => Promise<void>;
  reset: () => void;
  addLog: (type: LogEntry['type'], message: string, step?: number) => void;
  setStatus: (status: TaskStatus, step?: number) => void;
  setLocalisedFiles: (files: string[], graphData?: GraphData) => void;
  addAttempt: (attempt: AttemptRecord) => void;
  setResolved: (resolved: boolean, patch: string, tokens: number, elapsed: number) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL  = process.env.NEXT_PUBLIC_WS_URL  || 'ws://localhost:8000';

let logCounter = 0;
const makeId = () => `log_${++logCounter}_${Date.now()}`;

export const useAgentStore = create<AgentStore>((set, get) => ({
  // Form defaults
  repo: '',
  problemStatement: '',
  maxAttempts: 3,
  topKFiles: 5,
  setField: (key, value) => set({ [key]: value } as Partial<AgentStore>),

  // Execution state
  taskId: '',
  status: 'idle',
  step: 0,
  totalSteps: 5,

  // Results
  localisedFiles: [],
  attempts: [],
  currentPatch: '',
  resolved: false,
  failureCategory: '',
  totalTokens: 0,
  elapsedSeconds: 0,
  graphData: null,

  // Logs
  logs: [],

  // ── Actions ────────────────────────────────────────────────

  addLog: (type, message, step) => set(state => ({
    logs: [...state.logs.slice(-200), {
      id: makeId(),
      type,
      message,
      step,
      totalSteps: state.totalSteps,
      timestamp: new Date().toISOString(),
    }]
  })),

  setStatus: (status, step) => set(state => ({
    status,
    step: step ?? state.step,
  })),

  setLocalisedFiles: (files, graphData) => set({ localisedFiles: files, graphData: graphData ?? null }),

  addAttempt: (attempt) => set(state => ({
    attempts: [...state.attempts, attempt],
    currentPatch: attempt.patch,
    failureCategory: attempt.failureCategory,
  })),

  setResolved: (resolved, patch, tokens, elapsed) => set({
    resolved,
    currentPatch: patch,
    totalTokens: tokens,
    elapsedSeconds: elapsed,
  }),

  reset: () => set({
    taskId: '', status: 'idle', step: 0,
    localisedFiles: [], attempts: [], currentPatch: '',
    resolved: false, failureCategory: '',
    totalTokens: 0, elapsedSeconds: 0, graphData: null,
    logs: [],
  }),

  // ── Submit + WebSocket ─────────────────────────────────────

  submitJob: async () => {
    const { repo, problemStatement, maxAttempts, topKFiles, addLog, setStatus, reset } = get();

    if (!repo.trim() || !problemStatement.trim()) {
      addLog('error', 'Repository and problem statement are required.');
      return;
    }

    reset();
    setStatus('queued', 0);
    addLog('info', `Submitting job for ${repo}...`);

    // POST /api/solve
    let taskId: string;
    try {
      const res = await fetch(`${API_URL}/api/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo,
          problem_statement: problemStatement,
          max_attempts: maxAttempts,
          top_k_files: topKFiles,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      taskId = data.task_id;
      set({ taskId });
      addLog('info', `Task created: ${taskId}`);
    } catch (e) {
      addLog('error', `Failed to submit: ${(e as Error).message}`);
      setStatus('error');
      return;
    }

    // WebSocket streaming
    const ws = new WebSocket(`${WS_URL}/ws/${taskId}`);

    ws.onopen = () => {
      setStatus('running', 1);
      addLog('info', 'Connected to agent stream...');
    };

    ws.onmessage = (evt) => {
      let event: { event: string; data: Record<string, unknown>; timestamp: string };
      try { event = JSON.parse(evt.data); }
      catch { return; }

      const { addLog, setStatus, setLocalisedFiles, addAttempt, setResolved } = get();

      switch (event.event) {
        case 'log': {
          const d = event.data;
          const step = d.step as number | undefined;
          addLog('log', d.message as string, step);
          if (step) setStatus('running', step);
          break;
        }
        case 'localised_files': {
          const files = (event.data.files as string[]) || [];
          const graphData = {
            nodes: (event.data.graph_nodes as number) || 0,
            edges: (event.data.graph_edges as number) || 0,
          };
          setLocalisedFiles(files, graphData);
          addLog('success', `Localised ${files.length} files (${graphData.nodes} graph nodes)`);
          break;
        }
        case 'patch': {
          const d = event.data;
          addLog('log', `Attempt ${d.attempt}: patch generated`);
          set({ currentPatch: (d.patch as string) || '' });
          break;
        }
        case 'test_result': {
          const d = event.data;
          const resolved = d.resolved as boolean;
          addAttempt({
            attemptNum: d.attempt as number,
            patch: get().currentPatch,
            resolved,
            failureCategory: (d.failure_category as string) || '',
            failToPassResults: (d.fail_to_pass_results as Record<string, boolean>) || {},
          });
          addLog(
            resolved ? 'success' : 'warn',
            `Attempt ${d.attempt}: ${resolved ? '✅ Tests passed' : `❌ Tests failed (${d.failure_category})`}`
          );
          break;
        }
        case 'reflection': {
          const d = event.data;
          addLog('warn', `Reflecting on failure (${d.failure_category}) — attempt ${d.attempt}...`);
          break;
        }
        case 'done': {
          const d = event.data;
          setResolved(
            d.resolved as boolean,
            d.patch as string,
            (d.total_tokens as number) || 0,
            (d.elapsed_seconds as number) || 0,
          );
          setStatus('done');
          addLog(
            d.resolved ? 'success' : 'error',
            d.resolved
              ? `✅ Issue resolved in ${d.attempts} attempt(s)! (${d.elapsed_seconds}s)`
              : `❌ Could not resolve after ${d.attempts} attempt(s).`
          );
          ws.close();
          break;
        }
        case 'error': {
          const d = event.data;
          addLog('error', `Error: ${d.message}`);
          setStatus('error');
          ws.close();
          break;
        }
      }
    };

    ws.onerror = () => {
      addLog('error', 'WebSocket error — check that the API server is running.');
      setStatus('error');
    };

    ws.onclose = () => {
      if (get().status === 'running') {
        addLog('warn', 'Connection closed unexpectedly.');
        setStatus('error');
      }
    };
  },
}));
