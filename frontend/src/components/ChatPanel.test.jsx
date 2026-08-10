import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { STORAGE_KEY } from '../conversationStore'
import { askQuestion } from '../api'
import ChatPanel from './ChatPanel'


vi.mock('../api', () => ({
  askQuestion: vi.fn(),
  getApiErrorMessage: vi.fn((error, fallback) => error?.message || fallback),
}))


function answerPayload(answer, standaloneQuery, sources = []) {
  return {
    answer,
    sources,
    is_refused: false,
    conversation_context: {
      conversation_id: 'conversation-test',
      standalone_query: standaloneQuery,
      history_turn_count: 2,
      retained_turn_count: 2,
      compressed_turn_count: 0,
      was_compressed: false,
      summary_used: false,
      estimated_context_size: 120,
      query_rewrite_status: 'rewritten',
      compression_status: 'not_needed',
      fallback_used: false,
      context_limit_applied: false,
    },
  }
}


async function submitQuestion(question) {
  fireEvent.change(screen.getByLabelText('工业知识问题'), {
    target: { value: question },
  })
  fireEvent.click(screen.getByLabelText('提交问题'))
}


describe('multi-turn ChatPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders multiple messages in order and sends prior history on the second turn', async () => {
    askQuestion
      .mockResolvedValueOnce(answerPayload('PID 有三个环节。', '什么是 PID 控制？'))
      .mockResolvedValueOnce(
        answerPayload(
          '积分项用于消除稳态误差。',
          'PID 控制器中的积分项有什么作用？',
        ),
      )

    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('什么是 PID 控制？')
    expect(await screen.findByText('PID 有三个环节。')).toBeInTheDocument()

    await submitQuestion('其中积分项有什么作用？')
    expect(await screen.findByText('积分项用于消除稳态误差。')).toBeInTheDocument()

    const visibleMessages = screen
      .getAllByTestId('conversation-message')
      .map((element) => element.textContent)
    expect(visibleMessages).toEqual([
      '什么是 PID 控制？',
      expect.stringContaining('PID 有三个环节。'),
      '其中积分项有什么作用？',
      expect.stringContaining('积分项用于消除稳态误差。'),
    ])
    expect(askQuestion.mock.calls[1][0].history).toEqual([
      expect.objectContaining({ role: 'user', content: '什么是 PID 控制？' }),
      expect.objectContaining({ role: 'assistant', content: 'PID 有三个环节。' }),
    ])
    expect(askQuestion.mock.calls[1][0].conversationId).toMatch(
      /^conversation-/,
    )
  })

  it('restores messages from localStorage after refresh', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        schema_version: 1,
        active_conversation_id: 'conversation-restored',
        conversations: {
          'conversation-restored': {
            created_at: '2026-07-24T00:00:00.000Z',
            updated_at: '2026-07-24T00:01:00.000Z',
            messages: [
              {
                id: 'restored-message',
                role: 'user',
                content: '刷新后恢复的问题',
                timestamp: '2026-07-24T00:01:00.000Z',
              },
            ],
          },
        },
      }),
    )

    render(<ChatPanel modelProvider="Groq" />)
    expect(screen.getByText('刷新后恢复的问题')).toBeInTheDocument()
  })

  it('creates a new conversation and clears the visible thread', async () => {
    askQuestion.mockResolvedValueOnce(answerPayload('回答', '问题'))
    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('旧会话问题')
    expect(await screen.findByText('回答')).toBeInTheDocument()

    const firstId = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY),
    ).active_conversation_id
    fireEvent.click(screen.getByRole('button', { name: '新建会话' }))

    expect(screen.queryByText('旧会话问题')).not.toBeInTheDocument()
    const secondId = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY),
    ).active_conversation_id
    expect(secondId).not.toBe(firstId)
  })

  it('clears the active local record when clearing the conversation', async () => {
    askQuestion.mockResolvedValueOnce(answerPayload('回答', '问题'))
    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('待清空问题')
    expect(await screen.findByText('回答')).toBeInTheDocument()

    const oldId = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY),
    ).active_conversation_id
    fireEvent.click(screen.getByRole('button', { name: '清空当前会话' }))

    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY))
    expect(stored.conversations[oldId]).toBeUndefined()
    expect(screen.queryByText('待清空问题')).not.toBeInTheDocument()
  })

  it('shows optional context metadata in a collapsed details element', async () => {
    askQuestion.mockResolvedValueOnce(
      answerPayload(
        '积分项回答',
        'PID 控制器中的积分项有什么作用？',
      ),
    )
    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('其中积分项有什么作用？')
    await screen.findByText('积分项回答')

    const details = screen.getByText('上下文处理').closest('details')
    expect(details).not.toHaveAttribute('open')
    fireEvent.click(screen.getByText('上下文处理'))
    expect(
      screen.getByText('PID 控制器中的积分项有什么作用？', {
        exact: false,
      }),
    ).toBeInTheDocument()
    expect(screen.getByText('使用历史：2 条')).toBeInTheDocument()
    expect(screen.getByText('保留原文：2 条')).toBeInTheDocument()
    expect(screen.getByText('已压缩：0 条')).toBeInTheDocument()
  })

  it('keeps source rendering and tolerates a backend without conversation_context', async () => {
    askQuestion.mockResolvedValueOnce({
      answer: '兼容旧后端回答',
      sources: [
        { source: 'pid.pdf', page: 2, score: 0.1, content: '来源正文' },
      ],
      is_refused: false,
    })
    render(<ChatPanel modelProvider="DeepSeek" />)
    await submitQuestion('完整问题')

    expect(await screen.findByText('兼容旧后端回答')).toBeInTheDocument()
    expect(
      screen.getByText('[S1] pid.pdf · 第 2 页'),
    ).toBeInTheDocument()
    expect(screen.queryByText('上下文处理')).not.toBeInTheDocument()
  })

  it('preserves history after failure and retry does not duplicate the user message', async () => {
    askQuestion
      .mockRejectedValueOnce(new Error('临时失败'))
      .mockResolvedValueOnce(answerPayload('重试成功', 'PID 是什么？'))
    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('PID 是什么？')

    expect(await screen.findByText('临时失败')).toBeInTheDocument()
    expect(screen.getAllByText('PID 是什么？')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: '重新发送' }))

    expect(await screen.findByText('重试成功')).toBeInTheDocument()
    expect(screen.getAllByText('PID 是什么？')).toHaveLength(1)
  })

  it('prevents duplicate submission while a request is pending', async () => {
    let resolveRequest
    askQuestion.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRequest = resolve
      }),
    )
    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('只发送一次')

    expect(screen.getByLabelText('提交问题')).toBeDisabled()
    fireEvent.click(screen.getByLabelText('提交问题'))
    expect(askQuestion).toHaveBeenCalledTimes(1)

    resolveRequest(answerPayload('完成', '只发送一次'))
    await waitFor(() => expect(screen.getByText('完成')).toBeInTheDocument())
  })

  it('bounds a long assistant answer before sending it as history', async () => {
    const longAnswer = '长'.repeat(4500)
    askQuestion
      .mockResolvedValueOnce(answerPayload(longAnswer, '第一问'))
      .mockResolvedValueOnce(answerPayload('第二轮回答', '第二问'))

    render(<ChatPanel modelProvider="Groq" />)
    await submitQuestion('第一问')
    await screen.findByText(longAnswer)
    await submitQuestion('第二问')
    await screen.findByText('第二轮回答')

    const assistantHistory = askQuestion.mock.calls[1][0].history.find(
      (message) => message.role === 'assistant',
    )
    expect(assistantHistory.content).toHaveLength(4000)
  })

  it('limits the current question to the backend-compatible length', async () => {
    askQuestion.mockResolvedValueOnce(answerPayload('回答', '独立问题'))
    render(<ChatPanel modelProvider="Groq" />)
    expect(screen.getByLabelText('工业知识问题')).toHaveAttribute(
      'maxLength',
      '1000',
    )

    await submitQuestion('问'.repeat(1100))
    await screen.findByText('回答')
    expect(askQuestion.mock.calls[0][0].question).toHaveLength(1000)
  })
})
