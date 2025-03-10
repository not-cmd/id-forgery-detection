interface ResultDisplayProps {
  result: {
    is_forged: boolean
    confidence: number
    document_type: string
    visualization_path?: string
    original_image_path?: string
    forgery_indicators?: Record<string, any>
  }
}

const ResultDisplay = ({ result }: ResultDisplayProps) => {
  const { is_forged, confidence, document_type, visualization_path, forgery_indicators } = result
  
  if (!result || typeof is_forged === 'undefined') {
    return (
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-yellow-700">No valid result data available. Please try uploading an image again.</p>
          </div>
        </div>
      </div>
    )
  }
  
  const formatIndicatorName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const statusColor = is_forged ? 'bg-forged' : 'bg-genuine'
  const confidencePercentage = ((confidence || 0) * 100).toFixed(1)

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-100">
      <div className={`p-6 text-white ${statusColor}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold tracking-tight">
            {is_forged 
              ? `FORGED ${document_type?.toUpperCase() || 'DOCUMENT'} DETECTED` 
              : `GENUINE ${document_type?.toUpperCase() || 'DOCUMENT'}`}
          </h2>
          <div className="text-white/90 text-sm font-medium">
            Confidence: {confidencePercentage}%
          </div>
        </div>
      </div>
      
      <div className="p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Analysis Result</h3>
            <div className="relative aspect-video bg-gray-50 rounded-lg overflow-hidden border border-gray-200">
              {visualization_path ? (
                <img 
                  src={visualization_path} 
                  alt="Analysis Result" 
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    console.error('Error loading image:', visualization_path)
                    e.currentTarget.onerror = null
                    e.currentTarget.src = '/placeholder-image.svg'
                  }}
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="mt-2 text-sm text-gray-500">No visualization available</p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Confidence Analysis</h3>
            <div className="p-4 rounded-lg bg-gray-50 border border-gray-200">
              <div className="mb-2">
                <div className="flex justify-between text-sm font-medium">
                  <span>Confidence Score</span>
                  <span className={is_forged ? 'text-forged' : 'text-genuine'}>{confidencePercentage}%</span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div 
                    className={`h-2.5 rounded-full transition-all duration-500 ${is_forged ? 'bg-forged' : 'bg-genuine'}`}
                    style={{ width: `${confidencePercentage}%` }}
                  />
                </div>
              </div>
              
              <div className="mt-4 flex items-center justify-between text-sm">
                <span className="font-medium">Status</span>
                <span className={`px-2.5 py-0.5 rounded-full font-medium ${
                  is_forged 
                    ? 'bg-red-100 text-red-800' 
                    : 'bg-green-100 text-green-800'
                }`}>
                  {is_forged ? 'FORGED' : 'GENUINE'}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {forgery_indicators && Object.keys(forgery_indicators).length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Forensic Indicators</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(forgery_indicators)
                .filter(([key]) => key !== 'is_forged' && key !== 'forgery_score')
                .map(([key, value]) => (
                  <div key={key} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-sm font-medium text-gray-500">{formatIndicatorName(key)}</div>
                    <div className="mt-1 text-lg font-semibold text-gray-900">
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ResultDisplay 