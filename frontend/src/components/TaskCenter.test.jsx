import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import TaskCenter from './TaskCenter'

describe('TaskCenter', () => {
  afterEach(cleanup)

  it('shows worker health, failure details and retries a failed task', () => {
    const failedJob = {
      job_id: `job-${'a'.repeat(32)}`,
      task_type: 'build_draft',
      status: 'failed',
      progress: 60,
      message: '任务执行失败。',
      failed_stage: 'indexing',
      error: 'index failed',
      trace_id: `trace-${'a'.repeat(32)}`,
      created_at: '2026-07-28T06:00:00+00:00',
    }
    const onRetry = vi.fn()
    render(
      <TaskCenter
        center={{
          jobs: [failedJob],
          metrics: {
            total: 1,
            status_counts: { failed: 1 },
            p95_duration_seconds: 50,
          },
          worker: { healthy: false, backend: 'redis' },
        }}
        isLoading={false}
        retryingJobId=""
        adminToken="admin-secret"
        isManagementBusy={false}
        onRefresh={vi.fn()}
        onRetry={onRetry}
      />,
    )

    expect(screen.getByText('Worker 不可用')).toBeInTheDocument()
    expect(screen.getByText('indexing：index failed')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '重试任务' }))
    expect(onRetry).toHaveBeenCalledWith(failedJob)
  })
})
