import { useState, useRef, useCallback } from 'react'

interface DocumentUploaderProps {
  setResult: (result: any) => void
  setLoading: (loading: boolean) => void
}

const DocumentUploader = ({ setResult, setLoading }: DocumentUploaderProps) => {
  const [documentType, setDocumentType] = useState('id')
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.type.match('image.*')) {
      setError('Please upload an image file (jpg, jpeg, or png).')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size should not exceed 10MB.')
      return
    }

    setLoading(true)
    setResult(null)
    setError(null)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('document_type', documentType)

    try {
      console.log('Uploading file:', file.name, 'Type:', documentType)
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log('Response data:', data)
      setResult(data)
      setUploadProgress(100)
    } catch (error) {
      console.error('Error uploading file:', error)
      setError(`An error occurred while analyzing the image: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`)
      setResult(null)
    } finally {
      setLoading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }, [documentType, setLoading, setResult])

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
    <div className="w-full max-w-2xl mx-auto p-6">
      <div className="mb-6">
        <label className="block text-lg font-semibold text-gray-800 mb-3">Select Document Type:</label>
        <div className="flex space-x-6">
          <label className="relative inline-flex items-center cursor-pointer group">
            <input
              type="radio"
              className="peer sr-only"
              name="documentType"
              value="id"
              checked={documentType === 'id'}
              onChange={() => setDocumentType('id')}
            />
            <div className="p-4 bg-white border-2 border-gray-200 rounded-lg peer-checked:border-primary-500 peer-checked:bg-primary-50 transition-all duration-200 group-hover:border-primary-300">
              <span className="text-gray-700 peer-checked:text-primary-700">ID Card</span>
            </div>
          </label>
          <label className="relative inline-flex items-center cursor-pointer group">
            <input
              type="radio"
              className="peer sr-only"
              name="documentType"
              value="signature"
              checked={documentType === 'signature'}
              onChange={() => setDocumentType('signature')}
            />
            <div className="p-4 bg-white border-2 border-gray-200 rounded-lg peer-checked:border-primary-500 peer-checked:bg-primary-50 transition-all duration-200 group-hover:border-primary-300">
              <span className="text-gray-700 peer-checked:text-primary-700">Signature</span>
            </div>
          </label>
        </div>
      </div>

      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center
          transition-all duration-200 ease-in-out
          ${dragOver 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="text-5xl mb-4">📄</div>
        <p className="text-lg mb-4 text-gray-600">
          Drag and drop a document image here, or
          <button
            onClick={handleBrowseClick}
            className="text-primary-600 hover:text-primary-700 font-medium mx-1 focus:outline-none focus:underline"
          >
            browse
          </button>
          to select a file
        </p>
        <p className="text-sm text-gray-500">Supported formats: JPG, JPEG, PNG (max 10MB)</p>
        
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/jpeg,image/jpg,image/png"
          onChange={handleFileChange}
        />

        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>
      
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default DocumentUploader 