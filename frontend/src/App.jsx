import { useCallback, useEffect, useState } from 'react'

import {
  getApiErrorMessage,
  getHealth,
  getKnowledgeBaseJob,
  getKnowledgeBaseJobs,
  getKnowledgeBaseVersions,
  publishKnowledgeBase,
  resetKnowledgeBase,
  rollbackKnowledgeBaseVersion,
  retryKnowledgeBaseJob,
  uploadPdfs,
} from './api'
import { loadAdminToken, saveAdminToken } from './adminTokenStore'
import ChatPanel from './components/ChatPanel'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import SourcePanel from './components/SourcePanel'
import StudyTools from './components/StudyTools'

const initialHealth = {
  status: 'loading',
  knowledge_base_ready: false,
  pdf_count: 0,
}

const initialTaskCenter = {
  jobs: [],
  metrics: {},
  worker: {},
}

const TASK_POLL_INTERVAL_MS = 800
const TASK_WAIT_TIMEOUT_MS = 30 * 60 * 1000

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

async function waitForTask(job, adminToken, onProgress) {
  const deadline = Date.now() + TASK_WAIT_TIMEOUT_MS
  let current = job
  while (Date.now() < deadline) {
    current = await getKnowledgeBaseJob(current.job_id, adminToken)
    onProgress(current)
    if (current.status === 'succeeded') return current.result
    if (current.status === 'failed') {
      throw new Error(current.error || '后台任务执行失败。')
    }
    await wait(TASK_POLL_INTERVAL_MS)
  }
  throw new Error('后台任务等待超时，请稍后刷新任务状态。')
}

export default function App() {
  const [modelProvider, setModelProvider] = useState('Groq')
  const [adminToken, setAdminToken] = useState(() => loadAdminToken())
  const [health, setHealth] = useState(initialHealth)
  const [connectionError, setConnectionError] = useState('')
  const [files, setFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isResetting, setIsResetting] = useState(false)
  const [isPublishing, setIsPublishing] = useState(false)
  const [isLoadingVersions, setIsLoadingVersions] = useState(false)
  const [rollbackVersionId, setRollbackVersionId] = useState('')
  const [retryingJobId, setRetryingJobId] = useState('')
  const [versions, setVersions] = useState([])
  const [taskCenter, setTaskCenter] = useState(initialTaskCenter)
  const [isLoadingTasks, setIsLoadingTasks] = useState(false)
  const [uploadFeedback, setUploadFeedback] = useState(null)
  const [knowledgeBaseRevision, setKnowledgeBaseRevision] = useState(0)
  const [chatResult, setChatResult] = useState(null)

  const refreshHealth = useCallback(async () => {
    try {
      const payload = await getHealth()
      setHealth(payload)
      setConnectionError('')
    } catch (error) {
      setConnectionError(getApiErrorMessage(error, '无法连接 FastAPI 后端。'))
    }
  }, [])

  useEffect(() => {
    refreshHealth()
  }, [refreshHealth])

  const refreshVersions = useCallback(async (reportError = true) => {
    if (!adminToken.trim()) {
      setVersions([])
      return
    }
    setIsLoadingVersions(true)
    try {
      const payload = await getKnowledgeBaseVersions(adminToken)
      setVersions(payload.versions || [])
    } catch (error) {
      setVersions([])
      if (reportError) {
        setUploadFeedback({
          type: 'error',
          message: getApiErrorMessage(error, '知识库版本历史读取失败。'),
        })
      }
    } finally {
      setIsLoadingVersions(false)
    }
  }, [adminToken])

  const refreshTasks = useCallback(async (reportError = true) => {
    if (!adminToken.trim()) {
      setTaskCenter(initialTaskCenter)
      return
    }
    setIsLoadingTasks(true)
    try {
      const payload = await getKnowledgeBaseJobs(adminToken)
      setTaskCenter(payload)
    } catch (error) {
      if (reportError) {
        setUploadFeedback({
          type: 'error',
          message: getApiErrorMessage(error, '任务中心读取失败。'),
        })
      }
    } finally {
      setIsLoadingTasks(false)
    }
  }, [adminToken])

  const updateTaskRecord = useCallback((record) => {
    setTaskCenter((current) => ({
      ...current,
      jobs: [
        record,
        ...(current.jobs || []).filter(
          (item) => item.job_id !== record.job_id,
        ),
      ],
    }))
  }, [])

  useEffect(() => {
    if (!adminToken.trim()) return undefined
    const timeoutId = window.setTimeout(() => {
      refreshTasks(false)
    }, 600)
    return () => window.clearTimeout(timeoutId)
  }, [adminToken, refreshTasks])

  const hasActiveTasks = (taskCenter.jobs || []).some((job) =>
    ['pending', 'running'].includes(job.status),
  )

  useEffect(() => {
    if (!hasActiveTasks || !adminToken.trim()) return undefined
    const intervalId = window.setInterval(() => {
      refreshTasks(false)
    }, 2000)
    return () => window.clearInterval(intervalId)
  }, [adminToken, hasActiveTasks, refreshTasks])

  const handleBuild = async () => {
    if (files.length === 0) return

    setIsUploading(true)
    setUploadFeedback(null)
    try {
      const job = await uploadPdfs(files, adminToken)
      updateTaskRecord(job)
      const payload = await waitForTask(job, adminToken, (current) => {
        updateTaskRecord(current)
        setUploadFeedback({
          type: 'info',
          message: `${current.message} ${current.progress}%`,
        })
      })
      setUploadFeedback({
        type: 'success',
        message: `草稿库构建完成：${payload.page_count} 页，${payload.chunk_count} 个文本块。确认后请发布到公共知识库。`,
      })
      setFiles([])
    } catch (error) {
      setUploadFeedback({
        type: 'error',
        message: getApiErrorMessage(error, '草稿库构建失败。'),
      })
    } finally {
      setIsUploading(false)
      refreshTasks(false)
    }
  }

  const handleReset = async () => {
    if (!window.confirm('确定清空管理员草稿库吗？公共知识库不会受影响。')) return

    setIsResetting(true)
    try {
      await resetKnowledgeBase(adminToken)
      setFiles([])
      setUploadFeedback({ type: 'success', message: '草稿库已清空，公共知识库未受影响。' })
    } catch (error) {
      setUploadFeedback({
        type: 'error',
        message: getApiErrorMessage(error, '草稿库清空失败。'),
      })
    } finally {
      setIsResetting(false)
    }
  }

  const handlePublish = async () => {
    setIsPublishing(true)
    setUploadFeedback(null)
    try {
      const job = await publishKnowledgeBase(adminToken)
      updateTaskRecord(job)
      const payload = await waitForTask(job, adminToken, (current) => {
        updateTaskRecord(current)
        setUploadFeedback({
          type: 'info',
          message: `${current.message} ${current.progress}%`,
        })
      })
      setChatResult(null)
      setKnowledgeBaseRevision((revision) => revision + 1)
      setUploadFeedback({
        type: 'success',
        message: `公共知识库发布完成：${payload.page_count} 页，${payload.chunk_count} 个文本块。`,
      })
      await refreshHealth()
      await refreshVersions(false)
    } catch (error) {
      setUploadFeedback({
        type: 'error',
        message: getApiErrorMessage(error, '公共知识库发布失败。'),
      })
    } finally {
      setIsPublishing(false)
      refreshTasks(false)
    }
  }

  const handleRollbackVersion = async (version) => {
    if (
      !window.confirm(
        `确定将公共知识库回滚到版本 ${version.version_id} 吗？`,
      )
    ) return

    setRollbackVersionId(version.version_id)
    setUploadFeedback(null)
    try {
      const job = await rollbackKnowledgeBaseVersion(
        version.version_id,
        adminToken,
      )
      updateTaskRecord(job)
      const payload = await waitForTask(job, adminToken, (current) => {
        updateTaskRecord(current)
        setUploadFeedback({
          type: 'info',
          message: `${current.message} ${current.progress}%`,
        })
      })
      setChatResult(null)
      setKnowledgeBaseRevision((revision) => revision + 1)
      setUploadFeedback({
        type: 'success',
        message: `公共知识库已回滚到 ${payload.version_id}。`,
      })
      await refreshHealth()
      await refreshVersions(false)
    } catch (error) {
      setUploadFeedback({
        type: 'error',
        message: getApiErrorMessage(error, '公共知识库回滚失败。'),
      })
    } finally {
      setRollbackVersionId('')
      refreshTasks(false)
    }
  }

  const handleRetryTask = async (failedJob) => {
    setRetryingJobId(failedJob.job_id)
    setUploadFeedback(null)
    try {
      const job = await retryKnowledgeBaseJob(
        failedJob.job_id,
        adminToken,
      )
      updateTaskRecord(job)
      const payload = await waitForTask(job, adminToken, (current) => {
        updateTaskRecord(current)
        setUploadFeedback({
          type: 'info',
          message: `${current.message} ${current.progress}%`,
        })
      })
      setUploadFeedback({
        type: 'success',
        message: `任务重试成功：${job.task_type}。`,
      })
      if (['publish', 'rollback'].includes(job.task_type)) {
        setChatResult(null)
        setKnowledgeBaseRevision((revision) => revision + 1)
        await refreshHealth()
        await refreshVersions(false)
      } else if (payload?.files) {
        setFiles([])
      }
    } catch (error) {
      setUploadFeedback({
        type: 'error',
        message: getApiErrorMessage(error, '任务重试失败。'),
      })
    } finally {
      setRetryingJobId('')
      refreshTasks(false)
    }
  }

  return (
    <div className="app-shell">
      <Header
        modelProvider={modelProvider}
        onModelProviderChange={setModelProvider}
        health={health}
        connectionError={connectionError}
      />

      <div className="workspace-grid">
        <Sidebar
          modelProvider={modelProvider}
          onModelProviderChange={setModelProvider}
          health={health}
          connectionError={connectionError}
          files={files}
          onFilesChange={setFiles}
          onBuild={handleBuild}
          isUploading={isUploading}
          uploadFeedback={uploadFeedback}
          onReset={handleReset}
          isResetting={isResetting}
          onPublish={handlePublish}
          isPublishing={isPublishing}
          adminToken={adminToken}
          onAdminTokenChange={(value) => {
            setVersions([])
            setTaskCenter(initialTaskCenter)
            setAdminToken(saveAdminToken(value))
          }}
          versions={versions}
          isLoadingVersions={isLoadingVersions}
          rollbackVersionId={rollbackVersionId}
          onRefreshVersions={() => refreshVersions()}
          onRollbackVersion={handleRollbackVersion}
          taskCenter={taskCenter}
          isLoadingTasks={isLoadingTasks}
          retryingJobId={retryingJobId}
          onRefreshTasks={() => refreshTasks()}
          onRetryTask={handleRetryTask}
        />

        <main className="main-content" aria-label="AI 学习工作台">
          {connectionError && (
            <div className="alert error connection-alert" role="alert">
              {connectionError} 请先确认后端服务已启动或线上接口可访问。
            </div>
          )}

          <ChatPanel
            key={`chat-${knowledgeBaseRevision}`}
            modelProvider={modelProvider}
            onResult={setChatResult}
            knowledgeBaseRevision={knowledgeBaseRevision}
          />

          <StudyTools key={`study-${knowledgeBaseRevision}`} modelProvider={modelProvider} />
        </main>

        <SourcePanel sources={chatResult?.sources || []} />
      </div>
    </div>
  )
}
