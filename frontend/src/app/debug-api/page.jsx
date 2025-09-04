"use client"

import { useState } from 'react';

export default function DebugApiPage() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const addResult = (test, success, details) => {
    setResults(prev => [...prev, { test, success, details, timestamp: new Date().toLocaleTimeString() }]);
  };

  const runTests = async () => {
    setResults([]);
    setLoading(true);
    
    // Test 1: Environment variables
    addResult('Environment Check', true, {
      'NEXT_PUBLIC_API_URL': process.env.NEXT_PUBLIC_API_URL,
      'NODE_ENV': process.env.NODE_ENV,
      'Current URL': window.location.href
    });
    
    // Test 2: Different API base URLs
    const testUrls = [
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    ];
    
    for (const baseUrl of testUrls) {
      try {
        const response = await fetch(`${baseUrl}/api/health/`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        const data = await response.text();
        addResult(`Health Check (${baseUrl})`, response.ok, {
          status: response.status,
          statusText: response.statusText,
          data: data
        });
      } catch (error) {
        addResult(`Health Check (${baseUrl})`, false, {
          error: error.message,
          name: error.name,
          cause: error.cause?.message
        });
      }
    }
    
    // Test 3: Network info
    addResult('Network Info', true, {
      'User Agent': navigator.userAgent,
      'Online': navigator.onLine,
      'Connection': navigator.connection ? {
        effectiveType: navigator.connection.effectiveType,
        type: navigator.connection.type
      } : 'Not available'
    });
    
    setLoading(false);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">API Connectivity Diagnostics</h1>
      
      <div className="mb-6">
        <button 
          onClick={runTests}
          disabled={loading}
          className="bg-blue-500 text-white px-6 py-3 rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          {loading ? 'Running Tests...' : 'Run Diagnostic Tests'}
        </button>
      </div>

      <div className="space-y-4">
        {results.map((result, index) => (
          <div key={index} className={`p-4 rounded-lg border ${result.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">{result.test}</h3>
              <span className={`px-2 py-1 rounded text-xs ${result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {result.success ? 'PASS' : 'FAIL'}
              </span>
            </div>
            <div className="text-sm text-gray-600 mb-2">{result.timestamp}</div>
            <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto">
              {JSON.stringify(result.details, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}