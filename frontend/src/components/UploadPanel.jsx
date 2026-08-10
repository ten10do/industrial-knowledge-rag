import { useRef, useState } from 'react'
import { Database, FileText, UploadCloud, X } from 'lucide-react'

function getFileKey(file) {
  return `${file.name}-${file.size}-${file.lastModified}`
}

export default function UploadPanel({
  files,
  onFilesChange,
  onBuild,
  isUploading,
  feedback,
  adminToken,
  onAdminTokenChange,
  onPublish,
  isPublishing,
  isManagementBusy,
}) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationError, setValidationError] = useState('')

  const addFiles = (incomingFiles) => {
    const candidates = Array.from(incomingFiles)
    const pdfFiles = candidates.filter(
      (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'),
    )

    if (pdfFiles.length !== candidates.length) {
      setValidationError('仅支持上传 PDF 文件。')
    } else {
      setValidationError('')
    }

    const existingKeys = new Set(files.map(getFileKey))
    const nextFiles = [
      ...files,
      ...pdfFiles.filter((file) => !existingKeys.has(getFileKey(file))),
    ]
    onFilesChange(nextFiles)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    addFiles(event.dataTransfer.files)
  }

  const removeFile = (fileToRemove) => {
    const targetKey = getFileKey(fileToRemove)
    onFilesChange(files.filter((file) => getFileKey(file) !== targetKey))
  }

  return (
    <section className="sidebar-card" id="上传 PDF" aria-labelledby="upload-title">
      <div className="section-label-row">
        <UploadCloud size={18} aria-hidden="true" />
        <h2 id="upload-title">上传 PDF</h2>
      </div>

      <label className="field-label" htmlFor="admin-token">
        管理 Token
      </label>
      <input
        id="admin-token"
        className="admin-token-input"
        type="password"
        autoComplete="current-password"
        value={adminToken}
        onChange={(event) => onAdminTokenChange(event.target.value)}
        placeholder="构建、清空或发布时必填"
      />

      <div
        className={`upload-dropzone ${isDragging ? 'is-dragging' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <UploadCloud size={28} aria-hidden="true" />
        <p>拖拽工业知识 PDF 到此处</p>
        <span>支持一次上传多份技术资料</span>
        <button
          className="button button-secondary button-small"
          type="button"
          onClick={() => inputRef.current?.click()}
        >
          选择文件
        </button>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={(event) => {
            addFiles(event.target.files)
            event.target.value = ''
          }}
        />
      </div>

      {validationError && <p className="inline-error">{validationError}</p>}

      {files.length > 0 && (
        <ul className="file-list" aria-label="待上传文件">
          {files.map((file) => (
            <li key={getFileKey(file)}>
              <FileText size={16} aria-hidden="true" />
              <span title={file.name}>{file.name}</span>
              <button
                className="icon-button"
                type="button"
                title={`移除 ${file.name}`}
                aria-label={`移除 ${file.name}`}
                onClick={() => removeFile(file)}
              >
                <X size={15} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}

      <button
        className="button button-primary button-full"
        type="button"
        disabled={isManagementBusy || files.length === 0 || !adminToken.trim()}
        onClick={onBuild}
      >
        {isUploading ? <span className="spinner" aria-hidden="true" /> : <Database size={17} />}
        {isUploading ? '正在构建...' : '构建草稿库'}
      </button>

      <button
        className="button button-secondary button-full"
        type="button"
        disabled={isManagementBusy || !adminToken.trim()}
        onClick={onPublish}
      >
        {isPublishing && <span className="spinner dark" aria-hidden="true" />}
        {isPublishing ? '正在发布...' : '发布公共知识库'}
      </button>

      {feedback && (
        <p className={`feedback-message ${feedback.type}`} role="status">
          {feedback.message}
        </p>
      )}
    </section>
  )
}
