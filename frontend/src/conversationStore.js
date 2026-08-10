export const STORAGE_KEY = 'industrial-knowledge-rag-conversations-v1'
export const SCHEMA_VERSION = 1
export const MAX_CONVERSATIONS = 8
export const MAX_MESSAGES_PER_CONVERSATION = 60
export const MAX_HISTORY_MESSAGES = 40
export const MAX_HISTORY_CONTENT_CHARS = 4000
export const MAX_QUESTION_CONTENT_CHARS = 1000

function nowIso() {
  return new Date().toISOString()
}

export function createConversationId() {
  const suffix =
    globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
  return `conversation-${suffix}`
}

function createConversation(timestamp = nowIso()) {
  return {
    created_at: timestamp,
    updated_at: timestamp,
    messages: [],
  }
}

export function createEmptyConversationState(
  conversationId = createConversationId(),
) {
  return {
    schema_version: SCHEMA_VERSION,
    active_conversation_id: conversationId,
    conversations: {
      [conversationId]: createConversation(),
    },
  }
}

function normalizeSource(source) {
  if (!source || typeof source !== 'object') return null
  const normalized = {
    source: String(source.source || ''),
    page: Number(source.page || 0),
    score: Number(source.score || 0),
  }
  if (source.citation_id) normalized.citation_id = String(source.citation_id)
  return normalized
}

function normalizeMessage(message) {
  if (
    !message ||
    !['user', 'assistant'].includes(message.role) ||
    typeof message.content !== 'string' ||
    !message.content.trim()
  ) {
    return null
  }

  const normalized = {
    id: String(message.id || `message-${Date.now()}`),
    role: message.role,
    content: message.content.trim(),
    timestamp: String(message.timestamp || nowIso()),
  }
  if (message.role === 'assistant') {
    normalized.sources = Array.isArray(message.sources)
      ? message.sources.map(normalizeSource).filter(Boolean)
      : []
    normalized.is_refused = Boolean(message.is_refused)
    if (
      message.conversation_context &&
      typeof message.conversation_context === 'object'
    ) {
      normalized.conversation_context = message.conversation_context
    }
  }
  return normalized
}

function normalizeState(state) {
  if (
    !state ||
    state.schema_version !== SCHEMA_VERSION ||
    typeof state.active_conversation_id !== 'string' ||
    !state.conversations ||
    typeof state.conversations !== 'object'
  ) {
    return createEmptyConversationState()
  }

  const conversations = {}
  Object.entries(state.conversations).forEach(([id, conversation]) => {
    if (!conversation || typeof conversation !== 'object') return
    const messages = Array.isArray(conversation.messages)
      ? conversation.messages
          .map(normalizeMessage)
          .filter(Boolean)
          .slice(-MAX_MESSAGES_PER_CONVERSATION)
      : []
    conversations[id] = {
      created_at: String(conversation.created_at || nowIso()),
      updated_at: String(conversation.updated_at || nowIso()),
      messages,
    }
  })

  if (!conversations[state.active_conversation_id]) {
    conversations[state.active_conversation_id] = createConversation()
  }

  const ordered = Object.entries(conversations)
    .map((entry, index) => ({ entry, index }))
    .sort(
      (left, right) =>
        new Date(right.entry[1].updated_at).getTime() -
          new Date(left.entry[1].updated_at).getTime() ||
        right.index - left.index,
    )
    .map(({ entry }) => entry)
  const retained = ordered.slice(0, MAX_CONVERSATIONS)
  if (!retained.some(([id]) => id === state.active_conversation_id)) {
    retained[retained.length - 1] = [
      state.active_conversation_id,
      conversations[state.active_conversation_id],
    ]
  }

  return {
    schema_version: SCHEMA_VERSION,
    active_conversation_id: state.active_conversation_id,
    conversations: Object.fromEntries(retained),
  }
}

export function loadConversationState(storage = window.localStorage) {
  try {
    const serialized = storage.getItem(STORAGE_KEY)
    return serialized
      ? normalizeState(JSON.parse(serialized))
      : createEmptyConversationState()
  } catch {
    return createEmptyConversationState()
  }
}

export function saveConversationState(
  state,
  storage = window.localStorage,
) {
  const normalized = normalizeState(state)
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(normalized))
  } catch {
    // The current in-memory conversation remains usable if storage is full.
  }
  return normalized
}

export function startNewConversation(
  state,
  conversationId = createConversationId(),
) {
  const next = normalizeState(state)
  const timestamp = nowIso()
  return normalizeState({
    ...next,
    active_conversation_id: conversationId,
    conversations: {
      ...next.conversations,
      [conversationId]: createConversation(timestamp),
    },
  })
}

export function clearActiveConversation(
  state,
  replacementId = createConversationId(),
) {
  const next = normalizeState(state)
  const conversations = { ...next.conversations }
  delete conversations[next.active_conversation_id]
  return normalizeState({
    ...next,
    active_conversation_id: replacementId,
    conversations: {
      ...conversations,
      [replacementId]: createConversation(),
    },
  })
}

export function addMessage(
  state,
  message,
  conversationId = state.active_conversation_id,
) {
  const next = normalizeState(state)
  const normalizedMessage = normalizeMessage(message)
  if (!normalizedMessage) return next

  const existing = next.conversations[conversationId] || createConversation()
  return normalizeState({
    ...next,
    active_conversation_id: conversationId,
    conversations: {
      ...next.conversations,
      [conversationId]: {
        ...existing,
        updated_at: normalizedMessage.timestamp,
        messages: [...existing.messages, normalizedMessage].slice(
          -MAX_MESSAGES_PER_CONVERSATION,
        ),
      },
    },
  })
}
