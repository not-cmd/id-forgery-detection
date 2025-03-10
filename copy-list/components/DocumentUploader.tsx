import { useState, useRef } from 'react'

interface DocumentUploaderProps {
  setResult: (result: any) => void
  setLoading: (loading: boolean) => void
}

const DocumentUploader = ({ setResult, setLoading }: DocumentUploaderProps) => {
  const [documentType, setDocumentType] = useState('id')
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (file: File) => {
    if (!file.type.match('image.*')) {
      setError('Please upload an image file (jpg, jpeg, or png).')
      return
    }

    setLoading(true)
    setResult(null)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('document_type', documentType)

    try {
      console.log('Uploading file:', file.name, 'Type:', documentType)
      
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`)
      }

      console.log('Response data:', data)
      setResult(data)
    } catch (error) {
      console.error('Error uploading file:', error)
      setError(`An error occurred while analyzing the image: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`)
      setResult(null)
    } finally {
      setLoading(false)
      // Reset the file input so the same file can be uploaded again
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0])
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0])
    }
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="mb-8">
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Document Type:</label>
        <div className="flex space-x-4">
          <label className="inline-flex items-center">
            <input
              type="radio"
              className="form-radio text-blue-600"
              name="documentType"
              value="id"
              checked={documentType === 'id'}
              onChange={() => setDocumentType('id')}
            />
            <span className="ml-2">ID Card</span>
          </label>
          <label className="inline-flex items-center">
            <input
              type="radio"
              className="form-radio text-blue-600"
              name="documentType"
              value="signature"
              checked={documentType === 'signature'}
              onChange={() => setDocumentType('signature')}
            />
            <span className="ml-2">Signature</span>
          </label>
        </div>
      </div>

      <div
        className={`upload-area ${dragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="text-4xl mb-4">📄</div>
        <p className="mb-4">Drag and drop a document image here, or click to select a file</p>
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept=".jpg,.jpeg,.png"
          onChange={handleFileChange}
        />
        <button
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
          onClick={handleBrowseClick}
        >
          Browse Files
        </button>
      </div>
      
      {error && (
        <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
    </div>
  )
}

export default DocumentUploader 