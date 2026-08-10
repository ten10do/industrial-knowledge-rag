import { Activity, RefreshCw, RotateCcw } from 'lucide-react'

const TASK_LABELS = {
  build_draft: '构建草稿',
  publish: '发布公共库',
  rollback: '回滚版本',
}

const STATUS_LABELS = {
  pending: '排队中',
  running: '执行中',
  succeeded: '成功',
  failed: '失败',
}

function formatTime(value) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString('zh-CN', { hour12: false })
}

export default function TaskCenter({
  center,
  isLoading,
  retryingJobId,
  adminToken,
  isManagementBusy,
  onRefresh,
  onRetry,
}) {
  const jobs = center.jobs || []
  const metrics = center.metrics || {}
  const worker = center.worker || {}
  const counts = metrics.status_counts || {}

  return (
    <section className="sidebar-card task-center" aria-labelledby="task-center-title">
      <div className="section-label-row task-center-heading">
        <Activity size={18} aria-hidden="true" />
        <h2 id="task-center-title">任务中心</h2>
        <button
          className="icon-button"
          type="button"
          aria-label="刷新任务中心"
          title="刷新任务中心"
          disabled={isLoading || !adminToken.trim()}
          onClick={onRefresh}
        >
          <RefreshCw size={15} aria-hidden="true" />
        </button>
      </div>

      <div className="worker-status">
        <span className={worker.healthy ? 'worker-online' : 'worker-offline'}>
          {worker.healthy ? 'Worker 正常' : 'Worker 不可用'}
        </span>
        <span>{worker.backend || '未知队列'}</span>
      </div>

      <div className="task-metrics" aria-label="任务指标">
        <span>总计 {metrics.total || 0}</span>
        <span>成功 {counts.succeeded || 0}</span>
        <span>失败 {counts.failed || 0}</span>
        <span>
          均值 {metrics.average_duration_seconds == null
            ? '--'
            : `${metrics.average_duration_seconds}s`}
        </span>
        <span>
          P95 {metrics.p95_duration_seconds == null
            ? '--'
            : `${metrics.p95_duration_seconds}s`}
        </span>
      </div>

      {isLoading ? (
        <p className="status-meta">正在读取任务记录...</p>
      ) : jobs.length === 0 ? (
        <p className="status-meta">暂无后台任务。</p>
      ) : (
        <ul className="task-list">
          {jobs.map((job) => (
            <li key={job.job_id}>
              <div className="task-row">
                <strong>{TASK_LABELS[job.task_type] || job.task_type}</strong>
                <span className={`task-status ${job.status}`}>
                  {STATUS_LABELS[job.status] || job.status}
                </span>
              </div>
              <time dateTime={job.created_at}>{formatTime(job.created_at)}</time>
              <p>{job.message}</p>
              {['pending', 'running'].includes(job.status) && (
                <div
                  className="task-progress"
                  role="progressbar"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={job.progress}
                >
                  <span style={{ width: `${job.progress}%` }} />
                </div>
              )}
              {job.is_stalled && (
                <p className="task-warning">任务长时间无进度，请检查 Worker。</p>
              )}
              {job.status === 'failed' && (
                <>
                  <p className="task-error">
                    {job.failed_stage ? `${job.failed_stage}：` : ''}
                    {job.error || '未知错误'}
                  </p>
                  <button
                    className="button button-secondary button-small button-full"
                    type="button"
                    disabled={Boolean(retryingJobId) || isManagementBusy}
                    onClick={() => onRetry(job)}
                  >
                    {retryingJobId === job.job_id ? (
                      <span className="spinner dark" aria-hidden="true" />
                    ) : (
                      <RotateCcw size={14} aria-hidden="true" />
                    )}
                    {retryingJobId === job.job_id ? '正在重试...' : '重试任务'}
                  </button>
                </>
              )}
              <span className="task-trace" title={job.trace_id}>
                {job.trace_id}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
