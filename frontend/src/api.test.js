import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID } from './knowledgeBaseStore'

const { clientMock, createMock } = vi.hoisted(() => {
  const clientMock = {
    get: vi.fn(),
    post: vi.fn(),
  }
  return {
    clientMock,
    createMock: vi.fn(() => clientMock),
  }
})

vi.mock('axios', () => ({
  default: {
    create: createMock,
  },
}))

const {
  askQuestion,
  getApiErrorMessage,
  getKnowledgeBaseJob,
  getKnowledgeBaseJobs,
  getKnowledgeBaseVersions,
  publishKnowledgeBase,
  resetKnowledgeBase,
  rollbackKnowledgeBaseVersion,
  retryKnowledgeBaseJob,
  uploadPdfs,
} = await import('./api')

describe('API client configuration', () => {
  it('uses the local FastAPI server when no development URL is configured', () => {
    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ baseURL: 'http://localhost:8000' }),
    )
  })

  it('maps optional conversation fields onto the compatible ask payload', async () => {
    clientMock.post.mockResolvedValueOnce({ data: { answer: '回答' } })

    await askQuestion({
      question: '其中积分项有什么作用？',
      modelProvider: 'Groq',
      topK: 4,
      conversationId: 'conversation-api',
      history: [{ role: 'user', content: '什么是 PID？' }],
      contextOptions: { max_recent_turns: 4 },
    })

    expect(clientMock.post).toHaveBeenCalledWith(
      '/ask',
      {
        question: '其中积分项有什么作用？',
        model_provider: 'Groq',
        top_k: 4,
        conversation_id: 'conversation-api',
        history: [{ role: 'user', content: '什么是 PID？' }],
        context_options: { max_recent_turns: 4 },
      },
      {
        headers: {
          'X-Knowledge-Base-ID': DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID,
        },
      },
    )
  })

  it('formats FastAPI validation detail arrays as a stable message', () => {
    const message = getApiErrorMessage(
      {
        response: {
          data: {
            detail: [
              {
                loc: ['body', 'history', 0, 'content'],
                msg: 'String should have at most 4000 characters',
              },
            ],
          },
        },
      },
      '请求失败',
    )

    expect(message).toBe(
      'history.0.content：String should have at most 4000 characters',
    )
  })

  it('sends the management token only on mutating knowledge base requests', async () => {
    clientMock.post.mockResolvedValue({ data: {} })
    clientMock.get.mockResolvedValue({ data: { versions: [] } })

    await uploadPdfs([], 'admin-secret', 'upload-key-123')
    await resetKnowledgeBase('admin-secret')
    await publishKnowledgeBase('admin-secret', 'publish-key-123')
    await getKnowledgeBaseVersions('admin-secret')
    await rollbackKnowledgeBaseVersion(
      'v-20260728T060000000000Z-ffffffff',
      'admin-secret',
      'rollback-key-123',
    )
    await getKnowledgeBaseJob(`job-${'a'.repeat(32)}`, 'admin-secret')
    await getKnowledgeBaseJobs('admin-secret', 25)
    await retryKnowledgeBaseJob(
      `job-${'b'.repeat(32)}`,
      'admin-secret',
      'retry-key-123',
    )

    for (const call of clientMock.post.mock.calls.slice(-5)) {
      expect(call[2]).toEqual({
        headers: expect.objectContaining({
          'X-Admin-Token': 'admin-secret',
          'X-Knowledge-Base-ID': expect.stringMatching(/^kb-/),
        }),
      })
      expect(call[2].headers['X-Knowledge-Base-ID']).not.toBe(
        DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID,
      )
    }
    expect(
      clientMock.get.mock.calls.some((call) => call[0] === '/versions'),
    ).toBe(true)
    expect(clientMock.post.mock.calls.at(-2)[2].headers).toEqual(
      expect.objectContaining({
        'Idempotency-Key': 'rollback-key-123',
      }),
    )
    expect(clientMock.post.mock.calls.at(-2)[0]).toBe(
      '/versions/v-20260728T060000000000Z-ffffffff/rollback',
    )
    expect(clientMock.get.mock.calls.at(-1)).toEqual([
      '/jobs',
      expect.objectContaining({ params: { limit: 25 } }),
    ])
    expect(clientMock.post.mock.calls.at(-1)[0]).toBe(
      `/jobs/job-${'b'.repeat(32)}/retry`,
    )
  })
})
