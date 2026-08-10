import { FileSearch } from 'lucide-react'

import SourceCard from './SourceCard'

export default function SourcePanel({ sources }) {
  return (
    <aside className="source-panel" aria-labelledby="source-panel-title">
      <div className="source-panel-header">
        <div className="panel-icon small" aria-hidden="true">
          <FileSearch size={18} />
        </div>
        <div>
          <p className="eyebrow">SOURCE TRACE</p>
          <h2 id="source-panel-title">来源追溯</h2>
        </div>
      </div>

      <p className="source-panel-copy">
        回答生成后，这里会展示来源文件、页码、距离分数和参考片段。
      </p>

      {sources.length > 0 ? (
        <div className="source-list">
          {sources.map((source, index) => (
            <SourceCard
              key={`${source.source}-${source.page}-${index}`}
              source={source}
              index={index}
            />
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <FileSearch size={24} aria-hidden="true" />
          <span>暂无来源，提交问题后显示检索结果。</span>
        </div>
      )}
    </aside>
  )
}
