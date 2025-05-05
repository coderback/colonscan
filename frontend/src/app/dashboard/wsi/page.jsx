"use client"

import { useState } from 'react'
import axios from 'axios'

export default function WSIPage() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) return
    const formData = new FormData()
    formData.append('slide_file', file)
    setLoading(true)
    try {
      const { data } = await axios.post('/api/slides/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResult(data)
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex">
      <aside className="w-64 bg-white border-r flex flex-col justify-between">
        {/* Sidebar links moved to layout */}
      </aside>
      <main className="flex-1 overflow-auto bg-gray-50">
        <div className="p-6">
          <h1 className="text-3xl font-semibold mb-4">WSI Classification</h1>
          <p className="text-gray-600">Upload a whole-slide image to classify tissue type or detect abnormalities.</p>
        </div>
        <div className="p-6">
          <input
            type="file"
            accept=".svs,.tif,.tiff,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            className="mb-4"
          />
          <button
            onClick={handleUpload}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded"
          >
            {loading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
        {result && (
          <div className="p-6">
            <h2 className="text-2xl mb-2">Result</h2>
            <p>Status: {result.status}</p>
            {result.result_url && (
              <img src={result.result_url} alt="Heatmap" className="mt-4 max-w-full h-auto" />
            )}
          </div>
        )}
      </main>
    </div>
  )
}
