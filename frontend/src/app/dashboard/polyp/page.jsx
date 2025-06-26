"use client"

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { useAuth } from '@/context/AuthContext';
import UploadProgress from '@/components/UploadProgress';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Upload, Video, Play, Square, Clock, CheckCircle, AlertCircle } from 'lucide-react';

export default function PolypSegmentationPage() {
  const [tab, setTab] = useState("batch"); // 'batch' or 'streaming'
  const { token } = useAuth();
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoSessions, setVideoSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [streamingVideoFile, setStreamingVideoFile] = useState(null);
  const [streamingActive, setStreamingActive] = useState(false);
  const [streamingError, setStreamingError] = useState("");
  const [frameCount, setFrameCount] = useState(0);
  const streamingRef = useRef(null);
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionModalOpen, setSessionModalOpen] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const COLONOSCOPY_API = process.env.NEXT_PUBLIC_COLONOSCOPY_API_URL || 'http://localhost:8002';

  useEffect(() => {
    if (token) {
      fetchVideoSessions();
    }
    // Poll every 10s for job status updates
    const interval = setInterval(() => {
      if (token) fetchVideoSessions();
    }, 10000);
    return () => clearInterval(interval);
  }, [token]);

  const fetchVideoSessions = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE}/api/videosessions/`, {
        headers: { Authorization: `Token ${token}` }
      });
      setVideoSessions(res.data);
    } catch (err) {
      setError("Failed to fetch video sessions");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadProgress(0);
    setError("");
    const formData = new FormData();
    formData.append('video_file', selectedFile);
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);
      await axios.post(`${API_BASE}/api/videosessions/`, formData, {
        headers: { Authorization: `Token ${token}` }
      });
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(0), 1000);
      setSelectedFile(null);
      fetchVideoSessions();
    } catch (err) {
      setError("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'PENDING': return <Clock className="h-4 w-4 text-[#FFB81C]" />;
      case 'RUNNING': return <Play className="h-4 w-4 text-[#005EB8] animate-pulse" />;
      case 'COMPLETED': return <CheckCircle className="h-4 w-4 text-[#007F3B]" />;
      case 'FAILED': return <AlertCircle className="h-4 w-4 text-[#B00020]" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status) => {
    const base = "px-2.5 py-0.5 rounded-full text-xs font-medium";
    switch (status) {
      case 'PENDING': return `${base} bg-[#FFB81C]/10 text-[#FFB81C]`;
      case 'RUNNING': return `${base} bg-[#005EB8]/10 text-[#005EB8]`;
      case 'COMPLETED': return `${base} bg-[#007F3B]/10 text-[#007F3B]`;
      case 'FAILED': return `${base} bg-[#B00020]/10 text-[#B00020]`;
      default: return `${base} bg-gray-100 text-gray-800`;
    }
  };

  const handleStreamingSegmentation = async () => {
    if (!streamingVideoFile) return;
    
    setStreamingActive(true);
    setStreamingError("");
    setFrameCount(0);
    
    console.log("Starting streaming segmentation...");
    
    const formData = new FormData();
    formData.append('file', streamingVideoFile);
    
    try {
      const response = await fetch(`${COLONOSCOPY_API}/stream-segmentation`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Streaming failed');
      }
      
      console.log("Streaming response received, starting to read...");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let eventCount = 0;
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("Stream ended");
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete SSE events
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
              
              if (data.frame && data.image) {
                eventCount++;
                if (eventCount % 10 === 0) { // Log every 10th event
                  console.log(`Received frame ${data.frame}, image length: ${data.image.length}`);
                }
                
                setFrameCount(data.frame);
                
                // Create image from base64 data
                if (streamingRef.current) {
                  streamingRef.current.src = data.image;
                }
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', parseError, 'Line:', line);
            }
          }
        }
      }
    } catch (err) {
      console.error('Streaming error:', err);
      setStreamingError('Streaming failed: ' + err.message);
    } finally {
      setStreamingActive(false);
    }
  };

  const stopStreaming = () => {
    setStreamingActive(false);
    setFrameCount(0);
  };

  // Add a function to open the modal
  const openSessionModal = (session) => {
    setSelectedSession(session);
    setSessionModalOpen(true);
  };
  const closeSessionModal = () => {
    setSessionModalOpen(false);
    setSelectedSession(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#005EB8]"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Polyp Segmentation</h1>
          <p className="text-gray-600 mt-1">Upload and analyze colonoscopy videos for polyp detection</p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              tab === "batch" 
                ? "border-[#005EB8] text-[#005EB8]" 
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            }`}
            onClick={() => setTab("batch")}
          >
            Batch Video Segmentation
          </button>
          <button
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              tab === "streaming" 
                ? "border-[#005EB8] text-[#005EB8]" 
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            }`}
            onClick={() => setTab("streaming")}
          >
            Streaming Video Segmentation
          </button>
        </nav>
      </div>

      {/* Batch Video Segmentation */}
      {tab === "batch" && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <h2 className="text-lg font-bold text-neutral-800">Upload Colonoscopy Video</h2>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-[#005EB8] transition-colors">
                  <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                  <Input
                    type="file"
                    accept=".mp4"
                    onChange={handleFileChange}
                    className="hidden"
                    id="video-upload"
                    disabled={uploading}
                  />
                  <label htmlFor="video-upload" className="cursor-pointer">
                    <span className="text-sm text-gray-600">
                      {selectedFile ? selectedFile.name : 'Click to select video file'}
                    </span>
                  </label>
                </div>
                
                {uploading && (
                  <div className="space-y-2">
                    <Progress value={uploadProgress} className="w-full" />
                    <p className="text-xs text-gray-500 text-center">
                      {uploadProgress}% uploaded
                    </p>
                  </div>
                )}
                
                <Button
                  onClick={uploadVideo}
                  disabled={uploading || !selectedFile}
                  className="w-full"
                >
                  {uploading ? 'Uploading...' : 'Upload Video'}
                </Button>
                
                {error && (
                  <div className="flex items-center space-x-2 p-3 bg-[#B00020]/5 border border-[#B00020]/20 rounded-lg">
                    <AlertCircle className="h-4 w-4 text-[#B00020]" />
                    <span className="text-sm text-[#B00020]">{error}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h2 className="text-lg font-bold text-neutral-800">Video Sessions</h2>
            </CardHeader>
            <CardContent>
              {videoSessions.length === 0 ? (
                <div className="text-center py-8">
                  <Video className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No videos uploaded</h3>
                  <p className="text-gray-500">Upload your first video to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {videoSessions.map(session => (
                    <div key={session.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex-shrink-0">
                          <Video className="h-8 w-8 text-[#005EB8]" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900">
                            {session.video_file.split('/').pop()}
                          </p>
                          <p className="text-sm text-gray-500">
                            Uploaded: {new Date(session.uploaded).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(session.status)}
                          <span className={getStatusBadge(session.status)}>
                            {session.status}
                          </span>
                        </div>
                        {session.processed_video_url && session.status === 'COMPLETED' && (
                          <Button
                            asChild
                            variant="outline"
                            size="sm"
                          >
                            <a
                              href={session.processed_video_url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              Download Result
                            </a>
                          </Button>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openSessionModal(session)}
                        >
                          View Details
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
          {/* Session Modal */}
          {sessionModalOpen && selectedSession && (
            <div className="fixed inset-0 flex items-center justify-center z-50 p-4 pointer-events-auto">
              <div className="bg-white rounded-lg max-w-lg w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-gray-200">
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                  <h2 className="text-lg font-bold text-gray-900">Video Session Details</h2>
                  <Button variant="ghost" size="sm" onClick={closeSessionModal} className="text-gray-400 hover:text-gray-600">✕</Button>
                </div>
                <div className="p-4 space-y-4">
                  <div>
                    <span className="font-medium text-gray-700">Filename: </span>
                    <span className="text-gray-900">{selectedSession.video_file.split('/').pop()}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Status: </span>
                    <span className={getStatusBadge(selectedSession.status)}>{selectedSession.status}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Uploaded: </span>
                    <span className="text-gray-900">{new Date(selectedSession.uploaded).toLocaleString()}</span>
                  </div>
                  {selectedSession.processed_video_url && (
                    <div>
                      <span className="font-medium text-gray-700">Processed Video: </span>
                      <a href={selectedSession.processed_video_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">View/Download</a>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-end p-4 border-t border-gray-200">
                  <Button variant="outline" onClick={closeSessionModal}>Close</Button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Streaming Video Segmentation */}
      {tab === "streaming" && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <h2 className="text-lg font-bold text-neutral-800">Real-Time Video Streaming</h2>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-[#FFB81C] transition-colors">
                  <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                  <Input
                    type="file"
                    accept="video/mp4"
                    onChange={e => setStreamingVideoFile(e.target.files[0])}
                    className="hidden"
                    id="streaming-video-upload"
                    disabled={streamingActive}
                  />
                  <label htmlFor="streaming-video-upload" className="cursor-pointer">
                    <span className="text-sm text-gray-600">
                      {streamingVideoFile ? streamingVideoFile.name : 'Click to select video file'}
                    </span>
                  </label>
                </div>
                
                <div className="text-center">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Live Segmentation Output</h3>
                  <div className="relative inline-block">
                    <img
                      ref={streamingRef}
                      alt="Streaming Segmentation"
                      className="max-w-full rounded-lg border border-gray-200"
                      style={{ width: 480, height: 360, background: '#000' }}
                    />
                    {streamingActive && (
                      <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                        Frame: {frameCount}
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex space-x-2 justify-center">
                  <Button
                    onClick={handleStreamingSegmentation}
                    disabled={!streamingVideoFile || streamingActive}
                    className="bg-[#FFB81C] hover:bg-[#e6a600] text-white"
                  >
                    {streamingActive ? (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Streaming
                      </>
                    )}
                  </Button>
                  {streamingActive && (
                    <Button
                      onClick={stopStreaming}
                      variant="outline"
                      className="text-[#B00020] hover:bg-[#B00020]/10"
                    >
                      <Square className="h-4 w-4 mr-2" />
                      Stop
                    </Button>
                  )}
                </div>
                
                {streamingError && (
                  <div className="flex items-center space-x-2 p-3 bg-[#B00020]/5 border border-[#B00020]/20 rounded-lg">
                    <AlertCircle className="h-4 w-4 text-[#B00020]" />
                    <span className="text-sm text-[#B00020]">{streamingError}</span>
                  </div>
                )}
                
                <div className="text-center text-sm text-gray-600">
                  <p>This feature streams your video through the segmentation model in real-time,</p>
                  <p>showing each frame with polyp detection overlays as it's processed.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
