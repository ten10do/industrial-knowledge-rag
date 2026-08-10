import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  MAX_CONVERSATIONS,
  MAX_MESSAGES_PER_CONVERSATION,
  STORAGE_KEY,
  addMessage,
  clearActiveConversation,
  createConversationId,
  createEmptyConversationState,
  loadConversationState,
  saveConversationState,
  startNewConversation,
} from './conversationStore'


describe('conversation localStorage state', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('creates a versioned empty conversation state', () => {
    const state = createEmptyConversationState('conversation-first')

    expect(state.schema_version).toBe(1)
    expect(state.active_conversation_id).toBe('conversation-first')
    expect(state.conversations['conversation-first'].messages).toEqual([])
  })

  it('restores a saved conversation after refresh', () => {
    let state = createEmptyConversationState('conversation-restore')
    state = addMessage(state, {
      id: 'message-1',
      role: 'user',
      content: '什么是 PID 控制？',
      timestamp: '2026-07-24T00:00:00.000Z',
    })
    saveConversationState(state)

    const restored = loadConversationState()
    expect(restored.active_conversation_id).toBe('conversation-restore')
    expect(restored.conversations['conversation-restore'].messages).toHaveLength(1)
  })

  it('recovers from corrupted or incompatible localStorage', () => {
    window.localStorage.setItem(STORAGE_KEY, '{broken-json')
    const corrupted = loadConversationState()
    expect(corrupted.schema_version).toBe(1)
    expect(Object.keys(corrupted.conversations)).toHaveLength(1)

    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ schema_version: 99, conversations: {} }),
    )
    const incompatible = loadConversationState()
    expect(incompatible.schema_version).toBe(1)
    expect(Object.keys(incompatible.conversations)).toHaveLength(1)
  })

  it('evicts the oldest conversation when the local limit is exceeded', () => {
    let state = createEmptyConversationState('conversation-0')
    for (let index = 1; index <= MAX_CONVERSATIONS; index += 1) {
      state = startNewConversation(state, `conversation-${index}`)
    }

    expect(Object.keys(state.conversations)).toHaveLength(MAX_CONVERSATIONS)
    expect(state.conversations['conversation-0']).toBeUndefined()
    expect(state.active_conversation_id).toBe(`conversation-${MAX_CONVERSATIONS}`)
  })

  it('evicts the oldest messages without mixing conversations', () => {
    let state = createEmptyConversationState('conversation-one')
    for (let index = 0; index <= MAX_MESSAGES_PER_CONVERSATION; index += 1) {
      state = addMessage(state, {
        id: `message-${index}`,
        role: index % 2 === 0 ? 'user' : 'assistant',
        content: `消息 ${index}`,
        timestamp: new Date(index * 1000).toISOString(),
      })
    }
    state = startNewConversation(state, 'conversation-two')
    state = addMessage(state, {
      id: 'other-message',
      role: 'user',
      content: '另一个会话',
      timestamp: new Date().toISOString(),
    })

    expect(
      state.conversations['conversation-one'].messages,
    ).toHaveLength(MAX_MESSAGES_PER_CONVERSATION)
    expect(
      state.conversations['conversation-one'].messages[0].content,
    ).toBe('消息 1')
    expect(state.conversations['conversation-two'].messages[0].content).toBe(
      '另一个会话',
    )
  })

  it('clears the active conversation record and creates an isolated replacement', () => {
    let state = createEmptyConversationState('conversation-old')
    state = addMessage(state, {
      id: 'old-message',
      role: 'user',
      content: '旧消息',
      timestamp: new Date().toISOString(),
    })
    const cleared = clearActiveConversation(state, 'conversation-new')

    expect(cleared.conversations['conversation-old']).toBeUndefined()
    expect(cleared.active_conversation_id).toBe('conversation-new')
    expect(cleared.conversations['conversation-new'].messages).toEqual([])
  })

  it('creates IDs with the required conversation prefix', () => {
    expect(createConversationId()).toMatch(/^conversation-[A-Za-z0-9-]+$/)
  })
})
