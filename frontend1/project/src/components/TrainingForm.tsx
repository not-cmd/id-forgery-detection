import { useState, useRef } from 'react'

const TrainingForm = () => {
  const [documentType, setDocumentType] = useState('id')
  const [genuineFiles, setGenuineFiles] = useState<FileList | null>(null)
  const [forgedFiles, setForgedFiles] = useState<FileList | null>(null)
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const formRef = useRef<HTMLFormElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!genuineFiles || genuineFiles.length < 3) {
      alert('Please upload at least 3 genuine document images.')
      return
    }
    
    if (!forgedFiles || forgedFiles.length < 3) {
      alert('Please upload at least 3 forged document images.')
      return
    }
    
    setLoading(true)
    setSuccess(false)
    
    const formData = new FormData()
    formData.append('document_type', documentType)
    
    // Add genuine files
    for (let i = 0; i < genuineFiles.length; i++) {
      formData.append('genuine_files', genuineFiles[i])
    }
    
    // Add forged files
    for (let i = 0; i < forgedFiles.length; i++) {
      formData.append('forged_files', forgedFiles[i])
    }
    
    try {
      const response = await fetch('/train', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
      if (data.status === 'success') {
        setSuccess(true)
        formRef.current?.reset()
        setGenuineFiles(null)
        setForgedFiles(null)
      } else {
        alert('An error occurred: ' + data.error)
      }
    } catch (error) {
      console.error('Error starting training:', error)
      alert('An error occurred while starting the training process. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <h3 className="text-lg font-medium text-blue-700">Train Your Own Model</h3>
        <p className="text-blue-600">
          Upload your own genuine and forged documents to train a custom detection model. 
          This will help improve detection accuracy for your specific use case.
        </p>
      </div>
      
      <form ref={formRef} onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Document Type:</label>
          <div className="flex space-x-4">
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-blue-600"
                name="trainingDocumentType"
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
                name="trainingDocumentType"
                value="signature"
                checked={documentType === 'signature'}
                onChange={() => setDocumentType('signature')}
              />
              <span className="ml-2">Signature</span>
            </label>
          </div>
        </div>
        
        <div className="mb-6 p-4 border border-gray-200 rounded-lg">
          <h3 className="text-lg font-medium mb-2">Genuine Documents</h3>
          <p className="text-sm text-gray-600 mb-3">Upload images of genuine documents (at least 3 recommended).</p>
          <input
            type="file"
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            multiple
            accept=".jpg,.jpeg,.png"
            onChange={(e) => setGenuineFiles(e.target.files)}
          />
          {genuineFiles && (
            <div className="mt-2 text-sm text-gray-500">
              {genuineFiles.length} file(s) selected
            </div>
          )}
        </div>
        
        <div className="mb-6 p-4 border border-gray-200 rounded-lg">
          <h3 className="text-lg font-medium mb-2">Forged Documents</h3>
          <p className="text-sm text-gray-600 mb-3">Upload images of forged documents (at least 3 recommended).</p>
          <input
            type="file"
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            multiple
            accept=".jpg,.jpeg,.png"
            onChange={(e) => setForgedFiles(e.target.files)}
          />
          {forgedFiles && (
            <div className="mt-2 text-sm text-gray-500">
              {forgedFiles.length} file(s) selected
            </div>
          )}
        </div>
        
        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
          disabled={loading}
        >
          {loading ? 'Starting Training...' : 'Start Training'}
        </button>
      </form>
      
      {loading && (
        <div className="text-center my-6">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Starting training process. This may take some time...</p>
        </div>
      )}
      
      {success && (
        <div className="mt-6 bg-green-50 border-l-4 border-green-500 p-4">
          <h4 className="text-lg font-medium text-green-700">Training Started!</h4>
          <p className="text-green-600">
            The training process has been started in the background. This may take several minutes to complete 
            depending on the number of images provided.
          </p>
          <p className="text-green-600 mt-2">
            You can continue using the detection tab while training is in progress.
          </p>
        </div>
      )}
    </div>
  )
}

export default TrainingForm 