"use client"

import { useState } from 'react';
import axios from 'axios';

export default function TestApiPage() {
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const testHealthEndpoint = async () => {
    setLoading(true);
    try {
      const API_BASE = '/api/proxy';
      console.log('Testing API_BASE:', API_BASE);
      console.log('Full URL:', `${API_BASE}/health/`);
      
      const response = await axios.get(`${API_BASE}/health/`);
      setResult(`Success: ${JSON.stringify(response.data)}`);
      console.log('Health check success:', response.data);
    } catch (error) {
      setResult(`Error: ${error.message}`);
      console.error('Health check error:', error);
      console.error('Error response:', error.response?.data);
      console.error('Error status:', error.response?.status);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">API Connection Test</h1>
      
      <button 
        onClick={testHealthEndpoint}
        disabled={loading}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Testing...' : 'Test Health Endpoint'}
      </button>
      
      {result && (
        <div className="mt-4 p-4 border rounded">
          <pre className="whitespace-pre-wrap">{result}</pre>
        </div>
      )}
    </div>
  );
}