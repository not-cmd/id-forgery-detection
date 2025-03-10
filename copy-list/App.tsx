import { useState, useEffect } from 'react'
import './App.css'
import DocumentUploader from './components/DocumentUploader'
import ResultDisplay from './components/ResultDisplay'
import TrainingForm from './components/TrainingForm'

function App() {
  const [activeTab, setActiveTab] = useState('detection')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'loading' | 'connected' | 'error'>('loading')

  // Check if backend is running
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('/health')
        if (response.ok) {
          setBackendStatus('connected')
        } else {
          setBackendStatus('error')
        }
      } catch (error) {
        console.error('Error checking backend:', error)
        setBackendStatus('error')
      }
    }

    checkBackend()
  }, [])

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-3xl font-bold text-center mb-6">Document Forgery Detection</h1>
      
      {backendStatus === 'error' && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6">
          <p className="font-bold">Backend Connection Error</p>
          <p>Could not connect to the backend server. Please make sure it's running.</p>
        </div>
      )}
      
      {backendStatus === 'loading' && (
        <div className="text-center my-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Connecting to backend...</p>
        </div>
      )}
      
      {backendStatus === 'connected' && (
        <>
          <div className="mb-6">
            <div className="flex border-b border-gray-200">
              <button 
                className={`py-2 px-4 font-medium ${activeTab === 'detection' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                onClick={() => setActiveTab('detection')}
              >
                Detection
              </button>
              <button 
                className={`py-2 px-4 font-medium ${activeTab === 'training' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                onClick={() => setActiveTab('training')}
              >
                Training
              </button>
            </div>
          </div>
          
          {activeTab === 'detection' ? (
            <>
              <DocumentUploader 
                setResult={setResult} 
                setLoading={setLoading} 
              />
              {loading && (
                <div className="text-center my-8">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
                  <p className="mt-2 text-gray-600">Analyzing document...</p>
                </div>
              )}
              {result && <ResultDisplay result={result} />}
            </>
          ) : (
            <TrainingForm />
          )}
        </>
      )}
      
      <div className="mt-12 text-center text-sm text-gray-500">
        <p>Upload an image of an ID card or signature to check if it's genuine or forged.</p>
        <p>The system uses advanced computer vision and machine learning techniques to detect forgery.</p>
      </div>
    </div>
  )
}

export default App 