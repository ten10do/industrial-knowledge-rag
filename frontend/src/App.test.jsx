import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  askQuestion,
  getHealth,
  getKnowledgeBaseJob,
  getKnowledgeBaseJobs,
  getKnowledgeBaseVersions,
  publishKnowledgeBase,
  rollbackKnowledgeBaseVersion,
  retryKnowledgeBaseJob,
} from './api'

vi.mock('./api', () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: 'ok',
    knowledge_base_ready: true,
    pdf_count: 2,
  }),
  uploadPdfs: vi.fn(),
  askQuestion: vi.fn(),
  generateStudyContent: vi.fn(),
  getKnowledgeBaseJob: vi.fn(),
  getKnowledgeBaseJobs: vi.fn().mockResolvedValue({
    jobs: [],
    metrics: {},
    worker: { healthy: true, backend: 'memory' },
  }),
  getKnowledgeBaseVersions: vi.fn().mockResolvedValue({ versions: [] }),
  publishKnowledgeBase: vi.fn(),
  resetKnowledgeBase: vi.fn(),
  rollbackKnowledgeBaseVersion: vi.fn(),
  retryKnowledgeBaseJob: vi.fn(),
  getApiErrorMessage: vi.fn((error, fallback) => error?.message || fallback),
}))

describe('Industrial Knowledge RAG React application', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders the required frontend modules and backend status', async () => {
    render(<App />)

    expect(screen.getByText('Industrial Knowledge RAG')).toBeInTheDocument()
    expect(screen.getByText('工业知识智能检索与问答平台')).toBeInTheDocument()
    expect(screen.getAllByLabelText('模型选择')[0]).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '上传 PDF' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '与知识库对话' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '来源追溯' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '知识辅助' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '生成文档摘要' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '提取关键知识' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '生成核对问题' })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('知识库已就绪')).toBeInTheDocument()
    })
  })

  it('shows public version drift without hiding the loaded library', async () => {
    getHealth.mockResolvedValueOnce({
      status: 'degraded',
      knowledge_base_ready: true,
      pdf_count: 2,
      version_sync: {
        status: 'degraded',
        remote_active_version: 'version-2',
        loaded_version: 'version-1',
        last_error: 'download failed',
      },
    })

    render(<App />)

    expect(
      await screen.findByText('知识库版本同步降级'),
    ).toBeInTheDocument()
    expect(screen.getByText('已加载版本：version-1')).toBeInTheDocument()
    expect(screen.getByText('同步异常：download failed')).toBeInTheDocument()
  })

  it('distinguishes traffic governance degradation from version drift', async () => {
    getHealth.mockResolvedValueOnce({
      status: 'degraded',
      knowledge_base_ready: true,
      pdf_count: 2,
      version_sync: {
        status: 'synchronized',
        loaded_version: 'version-2',
      },
      governance: {
        rate_limit: { backend: 'redis', healthy: false },
        model_quota: { backend: 'redis', healthy: true },
      },
    })

    render(<App />)

    expect(
      await screen.findByText('限流或配额服务降级'),
    ).toBeInTheDocument()
    expect(
      screen.getByText('流量治理后端异常，已执行预设降级策略。'),
    ).toBeInTheDocument()
  })

  it('clears stale answers after a draft is published', async () => {
    askQuestion.mockResolvedValueOnce({
      answer: '当前知识库回答',
      sources: [],
      is_refused: false,
    })
    publishKnowledgeBase.mockResolvedValueOnce({
      job_id: `job-${'a'.repeat(32)}`,
      status: 'pending',
    })
    getKnowledgeBaseJob.mockResolvedValueOnce({
      job_id: `job-${'a'.repeat(32)}`,
      status: 'succeeded',
      progress: 100,
      message: '任务执行完成。',
      result: {
        page_count: 2,
        chunk_count: 4,
        files: ['course.pdf'],
      },
    })

    render(<App />)

    fireEvent.change(screen.getByLabelText('管理 Token'), {
      target: { value: 'admin-secret' },
    })
    fireEvent.change(screen.getByLabelText('工业知识问题'), {
      target: { value: '什么是 PLC 扫描周期？' },
    })
    fireEvent.click(screen.getByLabelText('提交问题'))

    expect(await screen.findByText('当前知识库回答')).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole('button', { name: '发布公共知识库' }),
    )

    await waitFor(() => {
      expect(screen.queryByText('当前知识库回答')).not.toBeInTheDocument()
    })
    expect(publishKnowledgeBase).toHaveBeenCalledWith('admin-secret')
  })

  it('loads version history and rolls the public library back', async () => {
    const olderVersion = {
      version_id: 'v-20260728T050000000000Z-eeeeeeee',
      created_at: '2026-07-28T05:00:00+00:00',
      page_count: 4,
      chunk_count: 8,
      files: ['older.pdf'],
      active: false,
    }
    getKnowledgeBaseVersions.mockResolvedValueOnce({
      versions: [olderVersion],
    })
    rollbackKnowledgeBaseVersion.mockResolvedValueOnce({
      job_id: `job-${'b'.repeat(32)}`,
      status: 'pending',
    })
    getKnowledgeBaseJob.mockResolvedValueOnce({
      job_id: `job-${'b'.repeat(32)}`,
      status: 'succeeded',
      progress: 100,
      message: '任务执行完成。',
      result: {
        ...olderVersion,
        files: ['older.pdf'],
      },
    })
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    render(<App />)
    fireEvent.change(screen.getByLabelText('管理 Token'), {
      target: { value: 'admin-secret' },
    })
    fireEvent.click(screen.getByRole('button', { name: '刷新版本历史' }))

    expect(
      await screen.findByText(olderVersion.version_id),
    ).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole('button', { name: '回滚到此版本' }),
    )

    await waitFor(() => {
      expect(rollbackKnowledgeBaseVersion).toHaveBeenCalledWith(
        olderVersion.version_id,
        'admin-secret',
      )
    })
  })
})
