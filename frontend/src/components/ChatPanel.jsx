import { useState } from 'react'
import {
  MessageSquareText,
  Plus,
  RotateCcw,
  Search,
  Send,
  Sparkles,
  Trash2,
} from 'lucide-react'

import { askQuestion, getApiErrorMessage } from '../api'
import {
  MAX_HISTORY_CONTENT_CHARS,
  MAX_HISTORY_MESSAGES,
  MAX_QUESTION_CONTENT_CHARS,
  addMessage,
  clearActiveConversation,
  createConversationId,
  createEmptyConversationState,
  loadConversationState,
  saveConversationState,
  startNewConversation,
} from '../conversationStore'

function createMessageId() {
  const suffix =
    globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
  return `message-${suffix}`
}

function historyPayload(messages) {
  return messages.slice(-MAX_HISTORY_MESSAGES).map((message) => ({
    role: message.role,
    content: message.content.slice(0, MAX_HISTORY_CONTENT_CHARS),
    timestamp: message.timestamp,
    ...(message.role === 'assistant' && message.sources?.length
      ? {
          sources: message.sources.map(({ source, page, score }) => ({
            source,
            page,
            score,
          })),
        }
      : {}),
  }))
}

function ContextDetails({ context }) {
  if (!context) return null
  return (
    <details className="context-details">
      <summary>上下文处理</summary>
      <dl>
        <div>
          <dt>独立检索问题</dt>
          <dd>改写结果：{context.standalone_query}</dd>
        </div>
        <div>
          <dt>历史处理</dt>
          <dd className="context-counts">
            <span>使用历史：{context.history_turn_count} 条</span>
            <span>保留原文：{context.retained_turn_count} 条</span>
            <span>已压缩：{context.compressed_turn_count} 条</span>
          </dd>
        </div>
        <div>
          <dt>处理状态</dt>
          <dd>
            {context.query_rewrite_status} · {context.compression_status}
            {context.fallback_used ? ' · 已使用安全回退' : ''}
          </dd>
        </div>
      </dl>
    </details>
  )
}

export default function ChatPanel({
  modelProvider,
  onResult,
  knowledgeBaseRevision = 0,
}) {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(4)
  const [conversationState, setConversationState] = useState(() => {
    const initial =
      knowledgeBaseRevision > 0
        ? createEmptyConversationState()
        : loadConversationState()
    return saveConversationState(initial)
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [failedMessageId, setFailedMessageId] = useState(null)

  const activeConversation =
    conversationState.conversations[
      conversationState.active_conversation_id
    ]
  const messages = activeConversation?.messages || []

  const persistState = (next) => {
    const persisted = saveConversationState(next)
    setConversationState(persisted)
    return persisted
  }

  const sendQuestion = async ({
    content,
    userMessageId,
    requestState,
    priorMessages,
  }) => {
    if (isLoading) return
    setError('')
    setFailedMessageId(null)
    setIsLoading(true)
    try {
      const payload = await askQuestion({
        question: content.slice(0, MAX_QUESTION_CONTENT_CHARS),
        modelProvider,
        topK,
        conversationId: requestState.active_conversation_id,
        history: historyPayload(priorMessages),
      })
      const withAnswer = addMessage(requestState, {
        id: createMessageId(),
        role: 'assistant',
        content: payload.answer,
        timestamp: new Date().toISOString(),
        sources: payload.sources || [],
        is_refused: payload.is_refused,
        conversation_context: payload.conversation_context,
      })
      persistState(withAnswer)
      onResult?.(payload)
    } catch (requestError) {
      setFailedMessageId(userMessageId)
      setError(getApiErrorMessage(requestError, '问答请求失败。'))
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const normalizedQuestion = question
      .trim()
      .slice(0, MAX_QUESTION_CONTENT_CHARS)
    if (!normalizedQuestion) {
      setError('请输入与已上传工业知识资料相关的问题。')
      return
    }
    if (isLoading) return

    const userMessage = {
      id: createMessageId(),
      role: 'user',
      content: normalizedQuestion,
      timestamp: new Date().toISOString(),
    }
    const priorMessages = messages
    const requestState = persistState(
      addMessage(conversationState, userMessage),
    )
    setQuestion('')
    await sendQuestion({
      content: normalizedQuestion,
      userMessageId: userMessage.id,
      requestState,
      priorMessages,
    })
  }

  const handleRetry = async () => {
    if (!failedMessageId || isLoading) return
    const messageIndex = messages.findIndex(
      (message) => message.id === failedMessageId,
    )
    if (messageIndex < 0) return
    await sendQuestion({
      content: messages[messageIndex].content,
      userMessageId: failedMessageId,
      requestState: conversationState,
      priorMessages: messages.slice(0, messageIndex),
    })
  }

  const handleNewConversation = () => {
    if (isLoading) return
    persistState(startNewConversation(conversationState))
    setQuestion('')
    setError('')
    setFailedMessageId(null)
    onResult?.(null)
  }

  const handleClearConversation = () => {
    if (isLoading) return
    persistState(clearActiveConversation(conversationState))
    setQuestion('')
    setError('')
    setFailedMessageId(null)
    onResult?.(null)
  }

  return (
    <section className="chat-workspace" aria-labelledby="chat-title">
      <div className="workspace-hero">
        <div>
          <p className="eyebrow">AI STUDY WORKSPACE</p>
          <h1 id="chat-title">与知识库对话</h1>
          <p>
            支持连续追问并自动压缩较早上下文，回答仍保留当前检索的来源文件、页码和距离分数。
          </p>
        </div>
        <div className="hero-actions">
          <div className="hero-chip">
            <Sparkles size={17} />
            <span>{modelProvider}</span>
          </div>
          <button
            type="button"
            className="secondary-button compact"
            onClick={handleNewConversation}
            disabled={isLoading}
          >
            <Plus size={15} />
            新建会话
          </button>
          <button
            type="button"
            className="secondary-button compact"
            onClick={handleClearConversation}
            disabled={isLoading}
          >
            <Trash2 size={15} />
            清空当前会话
          </button>
        </div>
      </div>

      <div className="chat-thread">
        {messages.length === 0 && !isLoading && (
          <div className="assistant-welcome">
            <div className="panel-icon" aria-hidden="true">
              <MessageSquareText size={21} />
            </div>
            <div>
              <h2>开始一次工业知识问答</h2>
              <p>
                例如：先询问“该设备的报警条件是什么？”，再追问“推荐的排查步骤有哪些？”
              </p>
            </div>
          </div>
        )}

        {messages.map((message) =>
          message.role === 'user' ? (
            <div
              className="message-row user"
              data-testid="conversation-message"
              key={message.id}
            >
              <div className="message-bubble user-bubble">
                {message.content}
              </div>
            </div>
          ) : (
            <div
              className="message-row assistant"
              data-testid="conversation-message"
              key={message.id}
            >
              <article
                className={`message-bubble assistant-card ${
                  message.is_refused ? 'refused' : ''
                }`}
              >
                <div className="answer-label">
                  <Search size={16} aria-hidden="true" />
                  {message.is_refused
                    ? '相关性不足'
                    : `${modelProvider} 回答`}
                </div>
                <div className="rich-text">{message.content}</div>
                {message.sources?.length > 0 && (
                  <div className="source-pills" aria-label="回答来源">
                    {message.sources.slice(0, 3).map((source, index) => (
                      <span
                        key={`${source.source}-${source.page}-${index}`}
                      >
                        [{source.citation_id || `S${index + 1}`}] {source.source}
                        {' · '}第 {source.page} 页
                      </span>
                    ))}
                    {message.sources.length > 3 && (
                      <span>+{message.sources.length - 3} 个来源</span>
                    )}
                  </div>
                )}
                <ContextDetails context={message.conversation_context} />
              </article>
            </div>
          ),
        )}

        {isLoading && (
          <div className="message-row assistant">
            <div className="message-bubble assistant-card loading-card">
              <span className="spinner dark" aria-hidden="true" />
              <span>正在检索知识库并生成回答...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="alert error conversation-error" role="alert">
            <span>{error}</span>
            {failedMessageId && (
              <button
                type="button"
                className="retry-button"
                onClick={handleRetry}
                disabled={isLoading}
              >
                <RotateCcw size={15} />
                重新发送
              </button>
            )}
          </div>
        )}
      </div>

      <form className="chat-composer" onSubmit={handleSubmit}>
        <label className="field-label" htmlFor="question-input">
          工业知识问题
        </label>
        <textarea
          id="question-input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="输入问题，或基于上一轮回答继续追问..."
          rows={3}
          maxLength={MAX_QUESTION_CONTENT_CHARS}
        />

        <div className="composer-footer">
          <label htmlFor="top-k">参考片段：{topK}</label>
          <input
            id="top-k"
            type="range"
            min="1"
            max="8"
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
          />
          <button
            className="send-button"
            type="submit"
            disabled={isLoading}
            aria-label="提交问题"
          >
            {isLoading ? (
              <span className="spinner" aria-hidden="true" />
            ) : (
              <Send size={19} />
            )}
          </button>
        </div>
      </form>
    </section>
  )
}
