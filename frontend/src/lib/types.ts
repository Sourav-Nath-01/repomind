// src/lib/types.ts
// ─────────────────
// Shared TypeScript types for the Code Review Agent UI

export type TaskStatus = 'idle' | 'queued' | 'running' | 'done' | 'error';

export interface SolveRequest {
  repo: string;
  problem_statement: string;
  instance_id?: string;
  base_commit?: string;
  fail_to_pass?: string[];
  pass_to_pass?: string[];
  max_attempts?: number;
  top_k_files?: number;
}

export interface WSEvent {
  event: 'status' | 'log' | 'localised_files' | 'patch' | 'test_result' | 'reflection' | 'done' | 'error';
  data: Record<string, unknown>;
  timestamp: string;
}

export interface LogEntry {
  id: string;
  type: 'info' | 'success' | 'error' | 'warn' | 'log';
  message: string;
  step?: number;
  totalSteps?: number;
  timestamp: string;
}

export interface AttemptRecord {
  attemptNum: number;
  patch: string;
  resolved: boolean;
  failureCategory: string;
  failToPassResults?: Record<string, boolean>;
}

export interface AgentResult {
  taskId: string;
  resolved: boolean;
  attempts: number;
  localisedFiles: string[];
  patch: string;
  failureCategory: string;
  totalTokens: number;
  elapsedSeconds: number;
}

export interface Metrics {
  total_issues_solved: number;
  avg_elapsed_seconds: number;
  avg_attempts: number;
  recall_at_5: number;
  total_token_cost: number;
  avg_token_cost_per_issue: number;
  failure_category_counts: Record<string, number>;
}

export interface GraphData {
  nodes: number;
  edges: number;
}
