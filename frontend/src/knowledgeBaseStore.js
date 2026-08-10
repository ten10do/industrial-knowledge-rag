export const KNOWLEDGE_BASE_STORAGE_KEY = 'industrial-knowledge-rag-knowledge-base-id-v1'
export const DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID =
  'kb-public-shared-00000001'

const KNOWLEDGE_BASE_ID_PATTERN = /^kb-[A-Za-z0-9_-]{16,64}$/
let inMemoryKnowledgeBaseId = ''

export function createKnowledgeBaseId() {
  const suffix =
    globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2, 18)}`
  return `kb-${suffix}`
}

export function getPublicKnowledgeBaseId() {
  const configured = String(
    import.meta.env.VITE_PUBLIC_KNOWLEDGE_BASE_ID ||
      DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID,
  ).trim()
  return KNOWLEDGE_BASE_ID_PATTERN.test(configured)
    ? configured
    : DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID
}

export function getKnowledgeBaseId(storage = window.localStorage) {
  try {
    const existing = storage.getItem(KNOWLEDGE_BASE_STORAGE_KEY)
    if (existing && KNOWLEDGE_BASE_ID_PATTERN.test(existing)) {
      inMemoryKnowledgeBaseId = existing
      return existing
    }

    const created = createKnowledgeBaseId()
    storage.setItem(KNOWLEDGE_BASE_STORAGE_KEY, created)
    inMemoryKnowledgeBaseId = created
    return created
  } catch {
    if (KNOWLEDGE_BASE_ID_PATTERN.test(inMemoryKnowledgeBaseId)) {
      return inMemoryKnowledgeBaseId
    }
    inMemoryKnowledgeBaseId = createKnowledgeBaseId()
    return inMemoryKnowledgeBaseId
  }
}
