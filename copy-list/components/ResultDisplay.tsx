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
      <div className="bg-yellow-100 border-l-4 border-yellow-500 p-4 mb-4">
        <p className="text-yellow-700">No valid result data available. Please try uploading an image again.</p>
      </div>
    )
  }
  
  const formatIndicatorName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className={`p-4 text-white ${is_forged ? 'forged' : 'genuine'}`}>
        <h2 className="text-xl font-bold">
          {is_forged 
            ? `FORGED ${document_type?.toUpperCase() || 'DOCUMENT'} DETECTED` 
            : `GENUINE ${document_type?.toUpperCase() || 'DOCUMENT'}`}
        </h2>
      </div>
      
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="text-lg font-medium mb-2">Analysis Result</h3>
            {visualization_path ? (
              <img 
                src={visualization_path} 
                alt="Analysis Result" 
                className="w-full rounded border border-gray-200"
                onError={(e) => {
                  console.error('Error loading image:', visualization_path)
                  e.currentTarget.onerror = null
                  e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2YxZjFmMSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiM5OTkiPkltYWdlIG5vdCBhdmFpbGFibGU8L3RleHQ+PC9zdmc+'
                }}
              />
            ) : (
              <div className="w-full h-40 bg-gray-100 flex items-center justify-center rounded border border-gray-200">
                <p className="text-gray-500">No visualization available</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-2">Confidence Score</h3>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className={`h-4 rounded-full ${is_forged ? 'bg-red-500' : 'bg-green-500'}`}
              style={{ width: `${(confidence || 0) * 100}%` }}
            ></div>
          </div>
          <p className="mt-1 text-sm text-gray-600">
            Confidence: {((confidence || 0) * 100).toFixed(2)}%
          </p>
        </div>
        
        {forgery_indicators && Object.keys(forgery_indicators).length > 0 && (
          <div>
            <h3 className="text-lg font-medium mb-2">Forensic Indicators</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(forgery_indicators)
                .filter(([key]) => key !== 'is_forged' && key !== 'forgery_score')
                .map(([key, value]) => (
                  <div key={key} className="bg-gray-50 p-3 rounded">
                    <strong>{formatIndicatorName(key)}:</strong>{' '}
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
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