import { describe, expect, it } from 'vitest'

import { loadAdminToken, saveAdminToken } from './adminTokenStore'

function memoryStorage() {
  const values = new Map()
  return {
    getItem: (key) => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  }
}

describe('admin token storage', () => {
  it('keeps the management token only in the provided session storage', () => {
    const storage = memoryStorage()

    expect(saveAdminToken('  secret  ', storage)).toBe('secret')
    expect(loadAdminToken(storage)).toBe('secret')
    expect(saveAdminToken('', storage)).toBe('')
    expect(loadAdminToken(storage)).toBe('')
  })
})
