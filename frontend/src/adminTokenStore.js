export const ADMIN_TOKEN_STORAGE_KEY = 'industrial-knowledge-rag-admin-token-v1'

export function loadAdminToken(storage = window.sessionStorage) {
  try {
    return storage.getItem(ADMIN_TOKEN_STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

export function saveAdminToken(token, storage = window.sessionStorage) {
  const normalized = String(token || '').trim()
  try {
    if (normalized) {
      storage.setItem(ADMIN_TOKEN_STORAGE_KEY, normalized)
    } else {
      storage.removeItem(ADMIN_TOKEN_STORAGE_KEY)
    }
  } catch {
    // The token remains available in React state for the current page.
  }
  return normalized
}
