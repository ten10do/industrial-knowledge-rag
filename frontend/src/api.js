import axios from 'axios'
import {
  getKnowledgeBaseId,
  getPublicKnowledgeBaseId,
} from './knowledgeBaseStore'

const apiClient = axios.create({
  baseURL:
    import.meta.env.VITE_API_BASE_URL ||
    (import.meta.env.DEV ? 'http://localhost:8000' : '/api'),
  timeout: 300000,
})

function scopedRequestConfig(
  config = {},
  knowledgeBaseId = getPublicKnowledgeBaseId(),
) {
  return {
    ...config,
    headers: {
      ...config.headers,
      'X-Knowledge-Base-ID': knowledgeBaseId,
    },
  }
}

function managementRequestConfig(adminToken, idempotencyKey = '') {
  const headers = {
    'X-Admin-Token': String(adminToken || '').trim(),
  }
  if (idempotencyKey) headers['Idempotency-Key'] = idempotencyKey
  return scopedRequestConfig(
    { headers },
    getKnowledgeBaseId(),
  )
}

export function createIdempotencyKey() {
  return (
    globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2, 18)}`
  )
}

function formatApiDetail(detail) {
  if (typeof detail === 'string') return detail
  if (!Array.isArray(detail)) return ''

  return detail
    .map((item) => {
      if (typeof item === 'string') return item
      if (!item || typeof item !== 'object') return ''
      const location = Array.isArray(item.loc)
        ? item.loc.filter((part) => part !== 'body').join('.')
        : ''
      const message = item.msg || item.message || ''
      return location && message ? `${location}：${message}` : message
    })
    .filter(Boolean)
    .join('；')
}

export function getApiErrorMessage(error, fallback = '请求失败，请稍后重试。') {
  return (
    formatApiDetail(error?.response?.data?.detail) ||
    error?.message ||
    fallback
  )
}

export async function getHealth() {
  const response = await apiClient.get('/health', scopedRequestConfig())
  return response.data
}

export async function uploadPdfs(
  files,
  adminToken,
  idempotencyKey = createIdempotencyKey(),
) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  const response = await apiClient.post(
    '/upload',
    formData,
    managementRequestConfig(adminToken, idempotencyKey),
  )
  return response.data
}

export async function askQuestion({
  question,
  modelProvider,
  topK,
  conversationId,
  history,
  contextOptions,
}) {
  const payload = {
    question,
    model_provider: modelProvider,
    top_k: topK,
  }
  if (conversationId) payload.conversation_id = conversationId
  if (Array.isArray(history)) payload.history = history
  if (contextOptions) payload.context_options = contextOptions

  const response = await apiClient.post(
    '/ask',
    {
      ...payload,
    },
    scopedRequestConfig(),
  )
  return response.data
}

export async function generateStudyContent(taskType, modelProvider) {
  const routes = {
    summary: '/study/summary',
    knowledge_points: '/study/knowledge-points',
    quiz: '/study/quiz',
  }

  const route = routes[taskType]
  if (!route) {
    throw new Error('不支持的学习辅助类型。')
  }

  const response = await apiClient.post(
    route,
    {
      model_provider: modelProvider,
    },
    scopedRequestConfig(),
  )
  return response.data
}

export async function resetKnowledgeBase(adminToken) {
  const response = await apiClient.post(
    '/reset',
    undefined,
    managementRequestConfig(adminToken),
  )
  return response.data
}

export async function publishKnowledgeBase(
  adminToken,
  idempotencyKey = createIdempotencyKey(),
) {
  const response = await apiClient.post(
    '/publish',
    undefined,
    managementRequestConfig(adminToken, idempotencyKey),
  )
  return response.data
}

export async function getKnowledgeBaseVersions(adminToken) {
  const response = await apiClient.get(
    '/versions',
    managementRequestConfig(adminToken),
  )
  return response.data
}

export async function rollbackKnowledgeBaseVersion(
  versionId,
  adminToken,
  idempotencyKey = createIdempotencyKey(),
) {
  const response = await apiClient.post(
    `/versions/${encodeURIComponent(versionId)}/rollback`,
    undefined,
    managementRequestConfig(adminToken, idempotencyKey),
  )
  return response.data
}

export async function getKnowledgeBaseJob(jobId, adminToken) {
  const response = await apiClient.get(
    `/jobs/${encodeURIComponent(jobId)}`,
    managementRequestConfig(adminToken),
  )
  return response.data
}

export async function getKnowledgeBaseJobs(adminToken, limit = 50) {
  const config = managementRequestConfig(adminToken)
  config.params = { limit }
  const response = await apiClient.get(
    '/jobs',
    config,
  )
  return response.data
}

export async function retryKnowledgeBaseJob(
  jobId,
  adminToken,
  idempotencyKey = createIdempotencyKey(),
) {
  const response = await apiClient.post(
    `/jobs/${encodeURIComponent(jobId)}/retry`,
    undefined,
    managementRequestConfig(adminToken, idempotencyKey),
  )
  return response.data
}
