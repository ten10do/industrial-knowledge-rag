import { ChevronDown, FileText } from 'lucide-react'

export default function SourceCard({ source, index }) {
  const score = Number.isFinite(source.score) ? source.score.toFixed(4) : 'N/A'
  const citationId = source.citation_id || `S${index + 1}`

  return (
    <details className="source-item">
      <summary>
        <div className="source-summary-main">
          <span className="source-index">[{citationId}]</span>
          <FileText size={17} aria-hidden="true" />
          <span className="source-title" title={source.source}>
            {source.source}
          </span>
        </div>
        <div className="source-tags">
          {source.equipment_model && <span>{source.equipment_model}</span>}
          {source.section && <span>{source.section}</span>}
          {source.knowledge_type && <span>{source.knowledge_type}</span>}
          <span>第 {source.page} 页</span>
          <span>距离 {score}</span>
        </div>
        <ChevronDown className="source-chevron" size={17} aria-hidden="true" />
      </summary>
      <div className="source-content">{source.content}</div>
    </details>
  )
}
