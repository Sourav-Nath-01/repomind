import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Autonomous Code Review & Bug-Fix Agent',
  description:
    'ML-powered agent that reads GitHub issues and generates patches — powered by ColBERT-v2 and GPT-4o-mini.',
  keywords: ['code review', 'bug fix', 'LLM agent', 'SWE-bench', 'AST parsing'],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body>{children}</body>
    </html>
  );
}
