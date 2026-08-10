import { Bot, Factory, Wifi, WifiOff } from 'lucide-react'

export default function Header({
  modelProvider,
  onModelProviderChange,
  health,
  connectionError,
}) {
  const isOnline = !connectionError && health.status !== 'loading'

  return (
    <header className="top-header">
      <div className="header-brand">
        <div className="header-logo" aria-hidden="true">
          <Factory size={24} />
        </div>
        <div>
          <strong>Industrial Knowledge RAG</strong>
          <span>工业知识智能检索与问答平台</span>
        </div>
      </div>

      <div className="header-actions">
        <div className={`backend-pill ${isOnline ? 'online' : 'offline'}`}>
          {isOnline ? <Wifi size={16} /> : <WifiOff size={16} />}
          <span>{isOnline ? 'Backend Online' : 'Backend Offline'}</span>
        </div>

        <label className="model-switcher" htmlFor="header-model-provider">
          <span>模型</span>
          <select
            id="header-model-provider"
            value={modelProvider}
            onChange={(event) => onModelProviderChange(event.target.value)}
          >
            <option value="DeepSeek">DeepSeek</option>
            <option value="Groq">Groq</option>
          </select>
        </label>

        <div className="avatar-badge" aria-label="工业知识助手">
          <Bot size={20} />
        </div>
      </div>
    </header>
  )
}
