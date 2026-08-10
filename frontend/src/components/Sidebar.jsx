import {
  BookOpenCheck,
  Brain,
  Cpu,
  Database,
  FileText,
  ListChecks,
  RotateCcw,
  Sparkles,
} from 'lucide-react'

import UploadPanel from './UploadPanel'
import VersionHistory from './VersionHistory'
import TaskCenter from './TaskCenter'

export default function Sidebar({
  modelProvider,
  onModelProviderChange,
  health,
  connectionError,
  files,
  onFilesChange,
  onBuild,
  isUploading,
  uploadFeedback,
  onReset,
  isResetting,
  onPublish,
  isPublishing,
  adminToken,
  onAdminTokenChange,
  versions,
  isLoadingVersions,
  rollbackVersionId,
  onRefreshVersions,
  onRollbackVersion,
  taskCenter,
  isLoadingTasks,
  retryingJobId,
  onRefreshTasks,
  onRetryTask,
}) {
  const isManagementBusy =
    isUploading ||
    isPublishing ||
    isResetting ||
    Boolean(retryingJobId) ||
    Boolean(rollbackVersionId)
  const versionSync = health.version_sync
  const governanceDegraded =
    health.governance &&
    (
      !health.governance.rate_limit?.healthy ||
      !health.governance.model_quota?.healthy
    )
  const statusLabel = connectionError
    ? '后端未连接'
    : versionSync?.status === 'degraded'
      ? '知识库版本同步降级'
      : governanceDegraded
        ? '限流或配额服务降级'
      : health.knowledge_base_ready
          ? '知识库已就绪'
          : '等待构建知识库'

  const navItems = [
    { label: '上传 PDF', icon: FileText, active: true },
    { label: '我的资料库', icon: Database },
    { label: '模型配置', icon: Cpu },
    { label: '文档摘要', icon: BookOpenCheck },
    { label: '知识点提取', icon: Brain },
    { label: '复习题生成', icon: ListChecks },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-card nav-card">
        <div className="sidebar-title">
          <Sparkles size={18} aria-hidden="true" />
          <span>知识库导航</span>
        </div>

        <nav className="nav-list" aria-label="知识库导航">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <a className={item.active ? 'active' : ''} href={`#${item.label}`} key={item.label}>
                <Icon size={17} aria-hidden="true" />
                <span>{item.label}</span>
              </a>
            )
          })}
        </nav>
      </div>

      <section className="sidebar-card" id="模型配置" aria-labelledby="model-title">
        <div className="section-label-row">
          <Cpu size={18} aria-hidden="true" />
          <h2 id="model-title">模型配置</h2>
        </div>
        <label className="field-label" htmlFor="model-provider">
          当前大模型
        </label>
        <select
          id="model-provider"
          aria-label="模型选择"
          value={modelProvider}
          onChange={(event) => onModelProviderChange(event.target.value)}
        >
          <option value="DeepSeek">DeepSeek</option>
          <option value="Groq">Groq</option>
        </select>
      </section>

      <UploadPanel
        files={files}
        onFilesChange={onFilesChange}
        onBuild={onBuild}
        isUploading={isUploading}
        feedback={uploadFeedback}
        adminToken={adminToken}
        onAdminTokenChange={onAdminTokenChange}
        onPublish={onPublish}
        isPublishing={isPublishing}
        isManagementBusy={isManagementBusy}
      />

      <VersionHistory
        versions={versions}
        isLoading={isLoadingVersions}
        rollbackVersionId={rollbackVersionId}
        adminToken={adminToken}
        isManagementBusy={isManagementBusy}
        onRefresh={onRefreshVersions}
        onRollback={onRollbackVersion}
      />

      <TaskCenter
        center={taskCenter}
        isLoading={isLoadingTasks}
        retryingJobId={retryingJobId}
        adminToken={adminToken}
        isManagementBusy={isManagementBusy}
        onRefresh={onRefreshTasks}
        onRetry={onRetryTask}
      />

      <section className="sidebar-card" aria-labelledby="status-title">
        <div className="section-label-row">
          <Database size={18} aria-hidden="true" />
          <h2 id="status-title">公共知识库状态</h2>
        </div>
        <div
          className={`status-line ${
            connectionError || health.status === 'degraded'
              ? 'offline'
              : 'online'
          }`}
        >
          <span className="status-dot" aria-hidden="true" />
          <span>{statusLabel}</span>
        </div>
        <p className="status-meta">已保存 PDF：{health.pdf_count ?? 0} 份</p>
        {versionSync?.loaded_version && (
          <p className="status-meta">已加载版本：{versionSync.loaded_version}</p>
        )}
        {versionSync?.last_error && (
          <p className="status-meta">同步异常：{versionSync.last_error}</p>
        )}
        {governanceDegraded && (
          <p className="status-meta">流量治理后端异常，已执行预设降级策略。</p>
        )}
        <button
          className="button button-danger button-full"
          type="button"
          disabled={
            isManagementBusy ||
            Boolean(connectionError) ||
            !adminToken.trim()
          }
          onClick={onReset}
        >
          {isResetting ? <span className="spinner" aria-hidden="true" /> : <RotateCcw size={17} />}
          {isResetting ? '正在清空...' : '清空草稿库'}
        </button>
      </section>

      <section className="sidebar-card steps" aria-labelledby="steps-title">
        <h2 id="steps-title">使用步骤</h2>
        <ol>
          <li>选择 DeepSeek 或 Groq 模型。</li>
          <li>管理员输入 Token 并上传课程 PDF。</li>
          <li>构建草稿并发布公共知识库。</li>
          <li>访客开始问答、追溯来源或生成学习资料。</li>
        </ol>
      </section>
    </aside>
  )
}
