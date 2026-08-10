import { describe, expect, it } from 'vitest'

import {
  DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID,
  KNOWLEDGE_BASE_STORAGE_KEY,
  getKnowledgeBaseId,
  getPublicKnowledgeBaseId,
} from './knowledgeBaseStore'

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial))
  return {
    getItem: (key) => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
  }
}

describe('knowledge base identity', () => {
  it('uses a stable public read-only knowledge base id', () => {
    expect(getPublicKnowledgeBaseId()).toBe(
      DEFAULT_PUBLIC_KNOWLEDGE_BASE_ID,
    )
  })

  it('persists one high-entropy scoped id per browser storage', () => {
    const storage = memoryStorage()
    const first = getKnowledgeBaseId(storage)
    const second = getKnowledgeBaseId(storage)

    expect(first).toMatch(/^kb-[A-Za-z0-9_-]{16,64}$/)
    expect(second).toBe(first)
  })

  it('replaces an invalid stored id', () => {
    const storage = memoryStorage({
      [KNOWLEDGE_BASE_STORAGE_KEY]: 'shared',
    })

    expect(getKnowledgeBaseId(storage)).toMatch(/^kb-/)
    expect(getKnowledgeBaseId(storage)).not.toBe('shared')
  })

  it('keeps one in-memory id when browser storage is unavailable', () => {
    const unavailableStorage = {
      getItem: () => {
        throw new Error('storage disabled')
      },
      setItem: () => {
        throw new Error('storage disabled')
      },
    }

    const first = getKnowledgeBaseId(unavailableStorage)
    expect(getKnowledgeBaseId(unavailableStorage)).toBe(first)
  })
})
