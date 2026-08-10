import { History, RefreshCw, RotateCcw } from 'lucide-react'

function formatCreatedAt(value) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString('zh-CN', { hour12: false })
}

export default function VersionHistory({
  versions,
  isLoading,
  rollbackVersionId,
  adminToken,
  isManagementBusy,
  onRefresh,
  onRollback,
}) {
  return (
    <section className="sidebar-card version-history" aria-labelledby="version-history-title">
      <div className="section-label-row version-history-heading">
        <History size={18} aria-hidden="true" />
        <h2 id="version-history-title">公共版本历史</h2>
        <button
          className="icon-button"
          type="button"
          aria-label="刷新版本历史"
          title="刷新版本历史"
          disabled={isLoading || isManagementBusy || !adminToken.trim()}
          onClick={onRefresh}
        >
          <RefreshCw size={15} aria-hidden="true" />
        </button>
      </div>

      {isLoading ? (
        <p className="status-meta">正在读取版本历史...</p>
      ) : versions.length === 0 ? (
        <p className="status-meta">暂无版本记录。</p>
      ) : (
        <ul className="version-list">
          {versions.map((version) => (
            <li key={version.version_id}>
              <div className="version-row">
                <span className="version-id" title={version.version_id}>
                  {version.version_id}
                </span>
                {version.active && <span className="version-active">当前</span>}
                {version.index_snapshot_ready && (
                  <span className="version-active">快照</span>
                )}
              </div>
              <time dateTime={version.created_at}>
                {formatCreatedAt(version.created_at)}
              </time>
              <p>
                {version.files.length} 份 PDF · {version.page_count} 页 ·{' '}
                {version.chunk_count} 个文本块
              </p>
              <button
                className="button button-secondary button-small button-full"
                type="button"
                disabled={
                  version.active ||
                  Boolean(rollbackVersionId) ||
                  isManagementBusy ||
                  !adminToken.trim()
                }
                onClick={() => onRollback(version)}
              >
                {rollbackVersionId === version.version_id ? (
                  <span className="spinner dark" aria-hidden="true" />
                ) : (
                  <RotateCcw size={14} aria-hidden="true" />
                )}
                {rollbackVersionId === version.version_id
                  ? '正在回滚...'
                  : version.active
                    ? '当前版本'
                    : '回滚到此版本'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
