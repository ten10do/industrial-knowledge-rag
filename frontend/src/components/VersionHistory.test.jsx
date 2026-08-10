import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import VersionHistory from './VersionHistory'

const versions = [
  {
    version_id: 'v-20260728T060000000000Z-ffffffff',
    created_at: '2026-07-28T06:00:00+00:00',
    page_count: 6,
    chunk_count: 12,
    files: ['current.pdf'],
    active: true,
    index_snapshot_ready: true,
  },
  {
    version_id: 'v-20260728T050000000000Z-eeeeeeee',
    created_at: '2026-07-28T05:00:00+00:00',
    page_count: 4,
    chunk_count: 8,
    files: ['older.pdf'],
    active: false,
  },
]

describe('VersionHistory', () => {
  afterEach(cleanup)

  it('marks the active version and allows rolling back an older version', () => {
    const onRollback = vi.fn()
    render(
      <VersionHistory
        versions={versions}
        isLoading={false}
        rollbackVersionId=""
        adminToken="admin-secret"
        onRefresh={vi.fn()}
        onRollback={onRollback}
      />,
    )

    expect(screen.getByRole('button', { name: '当前版本' })).toBeDisabled()
    expect(screen.getByText('快照')).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole('button', { name: '回滚到此版本' }),
    )
    expect(onRollback).toHaveBeenCalledWith(versions[1])
  })
})
