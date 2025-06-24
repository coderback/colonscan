"use client"

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { useAuth } from '@/context/AuthContext';
import UploadProgress from '@/components/UploadProgress';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Upload, Video, Play, Square } from 'lucide-react';
import { Video as VideoIcon } from 'lucide-react';

export default function PolypSegmentationPage() {
  const [tab, setTab] = useState("batch"); // 'batch', 'live', or 'streaming'
  const { token } = useAuth();
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoSessions, setVideoSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [liveFile, setLiveFile] = useState(null);
  const [liveUploading, setLiveUploading] = useState(false);
  const [liveProgress, setLiveProgress] = useState(0);
  const [liveResult, setLiveResult] = useState(null);
  const [liveError, setLiveError] = useState("");
  const [liveVideoFile, setLiveVideoFile] = useState(null);
  const [liveStreaming, setLiveStreaming] = useState(false);
  const [streamingVideoFile, setStreamingVideoFile] = useState(null);
  const [streamingActive, setStreamingActive] = useState(false);
  const [streamingError, setStreamingError] = useState("");
  const [frameCount, setFrameCount] = useState(0);
  const liveVideoRef = useRef(null);
  const liveResultRef = useRef(null);
  const streamingRef = useRef(null);

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

  const getStatusBadge = (status) => {
    const base = "px-2 py-1 rounded-full text-xs font-medium";
    switch (status) {
      case 'PENDING': return `${base} bg-yellow-100 text-yellow-800`;
      case 'RUNNING': return `${base} bg-blue-100 text-blue-800`;
      case 'COMPLETED': return `${base} bg-green-100 text-green-800`;
      case 'FAILED': return `${base} bg-red-100 text-red-800`;
      default: return `${base} bg-gray-100 text-gray-800`;
    }
  };

  const handleLiveUpload = async () => {
    if (!liveFile) return;
    setLiveUploading(true);
    setLiveProgress(0);
    setLiveError("");
    setLiveResult(null);
    const formData = new FormData();
    formData.append('file', liveFile);
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setLiveProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);
      const res = await fetch(`${COLONOSCOPY_API}/detect-frame`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Segmentation failed');
      const blob = await res.blob();
      setLiveResult(URL.createObjectURL(blob));
      setLiveProgress(100);
      setTimeout(() => setLiveProgress(0), 1000);
    } catch (err) {
      setLiveError('Segmentation failed. Please try again.');
    } finally {
      setLiveUploading(false);
    }
  };

  const handleStartLiveSegmentation = () => {
    if (!liveVideoFile) return;
    setLiveStreaming(true);
    setLiveError("");
    setTimeout(() => startStreaming(), 500); // slight delay to ensure video loads
  };

  const handleStopLiveSegmentation = () => {
    setLiveStreaming(false);
  };

  const startStreaming = () => {
    const video = liveVideoRef.current;
    if (!video) return;
    // Load video file into video element
    const url = URL.createObjectURL(liveVideoFile);
    video.src = url;
    video.load();
    video.play();
    // Start frame extraction loop
    let streaming = true;
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const sendFrame = async () => {
      if (!streaming || !video || video.paused || video.ended) return;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {
        if (!blob) return;
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        try {
          const res = await fetch(`${COLONOSCOPY_API}/detect-frame`, {
            method: 'POST',
            body: formData
          });
          if (res.ok) {
            const resultBlob = await res.blob();
            if (liveResultRef.current) {
              liveResultRef.current.src = URL.createObjectURL(resultBlob);
            }
          }
        } catch (err) {
          setLiveError('Segmentation failed.');
        }
      }, 'image/jpeg');
      // Next frame
      if (streaming && liveStreaming) {
        setTimeout(sendFrame, 200); // 5 fps
      }
    };
    // Start loop
    streaming = true;
    sendFrame();
    // Stop logic
    const stopListener = () => {
      streaming = false;
      video.removeEventListener('pause', stopListener);
      video.removeEventListener('ended', stopListener);
    };
    video.addEventListener('pause', stopListener);
    video.addEventListener('ended', stopListener);
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

  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-4">Polyp Segmentation</h1>
      <div className="mb-6 flex space-x-2 border-b border-gray-200">
        <button
          className={`px-4 py-2 font-medium border-b-2 transition-colors ${tab === "batch" ? "border-blue-600 text-blue-600" : "border-transparent text-gray-600 hover:text-blue-600"}`}
          onClick={() => setTab("batch")}
        >
          Batch Video Segmentation
        </button>
        <button
          className={`px-4 py-2 font-medium border-b-2 transition-colors ${tab === "live" ? "border-blue-600 text-blue-600" : "border-transparent text-gray-600 hover:text-blue-600"}`}
          onClick={() => setTab("live")}
        >
          Live Frame Segmentation
        </button>
        <button
          className={`px-4 py-2 font-medium border-b-2 transition-colors ${tab === "streaming" ? "border-blue-600 text-blue-600" : "border-transparent text-gray-600 hover:text-blue-600"}`}
          onClick={() => setTab("streaming")}
        >
          Streaming Video Segmentation
        </button>
      </div>
      {tab === "batch" && (
        <div className="space-y-6 max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Video className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Upload Colonoscopy Video</h2>
                  <p className="text-sm text-gray-600">Upload .mp4 files for batch polyp segmentation</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
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
              {error && <div className="mt-2 text-red-600 text-sm">{error}</div>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Video className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Previous Video Sessions</h2>
                  <p className="text-sm text-gray-600">History of your uploaded videos and segmentation results</p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-gray-500">Loading...</div>
              ) : videoSessions.length === 0 ? (
                <div className="text-center py-8">
                  <Video className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No videos uploaded</h3>
                  <p className="text-gray-500">Upload your first video to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {videoSessions.map(session => (
                    <div key={session.id} className="border border-gray-200 rounded-lg p-4 flex flex-col md:flex-row md:items-center md:justify-between hover:border-blue-300 transition-colors">
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-gray-900 truncate">{session.video_file.split('/').pop()}</div>
                        <div className="text-xs text-gray-600">Uploaded: {new Date(session.uploaded).toLocaleString()}</div>
                        <span className={getStatusBadge(session.status)}>{session.status}</span>
                      </div>
                      <div className="mt-2 md:mt-0 flex space-x-2">
                        {session.processed_video_url && session.status === 'COMPLETED' && (
                          <Button
                            asChild
                            variant="outline"
                            size="sm"
                            className="bg-green-600 text-white hover:bg-green-700"
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
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
      {tab === "live" && (
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <VideoIcon className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Live File-Based Segmentation</h2>
                  <p className="text-sm text-gray-600">Stream a video file and see segmentation overlays in real time</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-purple-400 transition-colors">
                <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <Input
                  type="file"
                  accept="video/mp4"
                  onChange={e => setLiveVideoFile(e.target.files[0])}
                  className="hidden"
                  id="live-video-upload"
                  disabled={liveUploading || liveStreaming}
                />
                <label htmlFor="live-video-upload" className="cursor-pointer">
                  <span className="text-sm text-gray-600">
                    {liveVideoFile ? liveVideoFile.name : 'Click to select video file'}
                  </span>
                </label>
              </div>
              <div className="flex flex-col md:flex-row md:space-x-6">
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Original Video</h3>
                  {liveVideoFile ? (
                    <video
                      ref={liveVideoRef}
                      controls
                      width={320}
                      height={240}
                      className="rounded border mx-auto"
                      style={{ background: '#000' }}
                    />
                  ) : (
                    <div className="text-center py-8 text-gray-400 border rounded-lg bg-gray-50">No video selected</div>
                  )}
                </div>
                <div className="flex-1 mt-6 md:mt-0">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Segmented Output</h3>
                  {liveVideoFile ? (
                    <img
                      ref={liveResultRef}
                      alt="Segmented Frame"
                      className="max-w-full rounded border mx-auto"
                      style={{ width: 320, height: 240, background: '#000' }}
                    />
                  ) : (
                    <div className="text-center py-8 text-gray-400 border rounded-lg bg-gray-50">No output yet</div>
                  )}
                </div>
              </div>
              <div className="flex space-x-2 justify-center">
                <Button
                  onClick={handleStartLiveSegmentation}
                  disabled={!liveVideoFile || liveUploading || liveStreaming}
                  className=""
                >
                  {liveStreaming ? "Streaming..." : "Start Live Segmentation"}
                </Button>
                {liveStreaming && (
                  <Button
                    onClick={handleStopLiveSegmentation}
                    variant="destructive"
                  >
                    Stop
                  </Button>
                )}
              </div>
              {liveUploading && (
                <div className="mt-4">
                  <Progress value={liveProgress} className="w-full" />
                </div>
              )}
              {liveError && <div className="mt-2 text-red-600 text-sm text-center">{liveError}</div>}
            </CardContent>
          </Card>
        </div>
      )}
      {tab === "streaming" && (
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-100 rounded-lg">
                  <Play className="h-5 w-5 text-orange-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Real-Time Video Streaming</h2>
                  <p className="text-sm text-gray-600">Upload a video and watch polyp segmentation happen in real-time</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-orange-400 transition-colors">
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
                    className="max-w-full rounded border"
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
                  className="bg-orange-600 hover:bg-orange-700"
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
                    variant="destructive"
                  >
                    <Square className="h-4 w-4 mr-2" />
                    Stop
                  </Button>
                )}
              </div>
              
              {streamingError && (
                <div className="mt-2 text-red-600 text-sm text-center">{streamingError}</div>
              )}
              
              <div className="text-center text-sm text-gray-600">
                <p>This feature streams your video through the segmentation model in real-time,</p>
                <p>showing each frame with polyp detection overlays as it's processed.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
