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
    <div className="min-h-screen bg-sky-100 py-8">
      <div className="container mx-auto px-4 max-w-5xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-sky-900 mb-2">Document Forgery Detection</h1>
          <p className="text-lg text-sky-700">
            Upload an ID card or signature to check if it's genuine or forged
          </p>
        </header>
        
        {backendStatus === 'error' && (
          <div className="mb-8 bg-red-50 border-l-4 border-red-400 p-4 rounded-r-lg">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Backend Connection Error</h3>
                <p className="mt-1 text-sm text-red-700">Could not connect to the backend server. Please make sure it's running.</p>
              </div>
            </div>
          </div>
        )}
        
        {backendStatus === 'loading' && (
          <div className="flex items-center justify-center py-12">
            <div className="animate-pulse-slow flex flex-col items-center">
              <div className="rounded-full h-12 w-12 border-4 border-sky-500 border-t-transparent animate-spin"></div>
              <p className="mt-4 text-sky-700">Connecting to backend...</p>
            </div>
          </div>
        )}
        
        {backendStatus === 'connected' && (
          <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg overflow-hidden border border-sky-200">
            <div className="border-b border-sky-200">
              <nav className="flex justify-center -mb-px" aria-label="Tabs">
                <button 
                  className={`
                    py-4 px-8 font-medium text-sm border-b-2 transition-colors duration-200
                    ${activeTab === 'detection'
                      ? 'border-sky-500 text-sky-600'
                      : 'border-transparent text-sky-500 hover:text-sky-700 hover:border-sky-300'
                    }
                  `}
                  onClick={() => setActiveTab('detection')}
                >
                  Detection
                </button>
                <button 
                  className={`
                    py-4 px-8 font-medium text-sm border-b-2 transition-colors duration-200
                    ${activeTab === 'training'
                      ? 'border-sky-500 text-sky-600'
                      : 'border-transparent text-sky-500 hover:text-sky-700 hover:border-sky-300'
                    }
                  `}
                  onClick={() => setActiveTab('training')}
                >
                  Training
                </button>
              </nav>
            </div>
            
            <div className="p-6">
              {activeTab === 'detection' ? (
                <div className="space-y-8">
                  <DocumentUploader 
                    setResult={setResult} 
                    setLoading={setLoading} 
                  />
                  {loading && (
                    <div className="flex items-center justify-center py-12">
                      <div className="animate-pulse-slow flex flex-col items-center">
                        <div className="rounded-full h-12 w-12 border-4 border-sky-500 border-t-transparent animate-spin"></div>
                        <p className="mt-4 text-sky-700">Analyzing document...</p>
                      </div>
                    </div>
                  )}
                  {result && <ResultDisplay result={result} />}
                </div>
              ) : (
                <TrainingForm />
              )}
            </div>
          </div>
        )}
        
        <footer className="mt-12 text-center text-sm text-sky-600 space-y-2">
          <p>Our system uses advanced computer vision and machine learning techniques to detect document forgery.</p>
          <p>Supported document types: ID cards and signatures.</p>
        </footer>
      </div>
    </div>
  )
}

export default App 