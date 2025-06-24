"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '@/context/AuthContext';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import SlideDetailModal from '@/components/SlideDetailModal';
import UploadProgress from '@/components/UploadProgress';
import { 
  Upload, 
  Microscope, 
  FileText, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Play,
  Eye,
  Download,
  Trash2,
  RefreshCw,
  BarChart3,
  Image as ImageIcon
} from 'lucide-react';

export default function WSIPage() {
  const { token } = useAuth();
  const [slides, setSlides] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedSlide, setSelectedSlide] = useState(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [patchFiles, setPatchFiles] = useState([]);
  const [patchUploading, setPatchUploading] = useState(false);
  const [patchProgress, setPatchProgress] = useState(0);
  const [patchResults, setPatchResults] = useState([]);
  const [patchError, setPatchError] = useState(null);

  // API base URL
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    if (token) {
      fetchSlides();
    }
  }, [token]);

  // Auto-refresh slides every 10 seconds to check for status updates
  useEffect(() => {
    if (token) {
      const interval = setInterval(fetchSlides, 10000);
      return () => clearInterval(interval);
    }
  }, [token]);

  const fetchSlides = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/api/slides/`, {
        headers: { Authorization: `Token ${token}` }
      });
      setSlides(response.data);
    } catch (error) {
      console.error('Failed to fetch slides:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['.svs', '.tiff', '.tif'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      
      if (!validTypes.includes(fileExtension)) {
        alert('Please select a valid WSI file (.svs, .tiff, .tif)');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const uploadSlide = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('slide_file', selectedFile);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      await axios.post(`${API_BASE}/api/slides/`, formData, {
        headers: {
          Authorization: `Token ${token}`
        }
      });

      setUploadProgress(100);
      setTimeout(() => {
        setUploadProgress(0);
        setSelectedFile(null);
        fetchSlides();
      }, 1000);

    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const deleteSlide = async (slideId) => {
    if (!confirm('Are you sure you want to delete this slide?')) return;

    try {
      await axios.delete(`${API_BASE}/api/slides/${slideId}/`, {
        headers: { Authorization: `Token ${token}` }
      });
      fetchSlides();
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Delete failed. Please try again.');
    }
  };

  const openDetailModal = (slide) => {
    setSelectedSlide(slide);
    setDetailModalOpen(true);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'PENDING':
        return <Clock className="h-4 w-4 text-yellow-600" />;
      case 'RUNNING':
        return <Play className="h-4 w-4 text-blue-600 animate-pulse" />;
      case 'COMPLETED':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'FAILED':
        return <AlertCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusBadge = (status) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium";
    switch (status) {
      case 'PENDING':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'RUNNING':
        return `${baseClasses} bg-blue-100 text-blue-800`;
      case 'COMPLETED':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'FAILED':
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getFileName = (filePath) => {
    return filePath.split('/').pop();
  };

  // Patch upload handler
  const handlePatchChange = (e) => {
    setPatchFiles(Array.from(e.target.files));
  };

  const uploadPatches = async () => {
    if (!patchFiles.length) return;
    setPatchUploading(true);
    setPatchProgress(0);
    setPatchError(null);
    setPatchResults([]);
    const form = new FormData();
    patchFiles.forEach(file => form.append('files', file));
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setPatchProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);
      const res = await axios.post(`${API_BASE}/api/patches/batch/`, form, {
        headers: {
          Authorization: `Token ${token}`
        }
      });
      setPatchProgress(100);
      setTimeout(() => setPatchProgress(0), 1000);
      setPatchResults(res.data);
    } catch (err) {
      setPatchError('Patch upload failed.');
      console.error('Patch upload error:', err);
    } finally {
      setPatchUploading(false);
      setPatchFiles([]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="border-b border-gray-200 pb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Microscope className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Whole Slide Image Analysis</h1>
              <p className="text-gray-600 mt-1">Upload and analyze histopathology slides with AI</p>
            </div>
          </div>
          <Button 
            onClick={fetchSlides} 
            variant="outline" 
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Upload className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Upload Slide</h2>
                  <p className="text-sm text-gray-600">Upload .svs or .tiff files</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <Input 
                  type="file" 
                  accept=".svs,.tiff,.tif" 
                  onChange={handleFileSelect}
                  className="hidden"
                  id="slide-upload"
                />
                <label htmlFor="slide-upload" className="cursor-pointer">
                  <span className="text-sm text-gray-600">
                    {selectedFile ? selectedFile.name : 'Click to select WSI file'}
                  </span>
                </label>
              </div>
              
              {uploadProgress > 0 && (
                <div className="space-y-2">
                  <Progress value={uploadProgress} className="w-full" />
                  <p className="text-xs text-gray-500 text-center">
                    {uploadProgress}% uploaded
                  </p>
                </div>
              )}
              
              <Button 
                onClick={uploadSlide} 
                disabled={uploading || !selectedFile}
                className="w-full"
              >
                {uploading ? 'Uploading...' : 'Upload Slide'}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Patch Analysis Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <FileText className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Patch Analysis</h2>
                  <p className="text-sm text-gray-600">Upload tissue patches for classification</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-green-400 transition-colors cursor-pointer"
                onClick={() => document.getElementById('patch-upload').click()}
                tabIndex={0}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('patch-upload').click(); }}
                role="button"
                aria-label="Select patch images"
              >
                <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <Input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handlePatchChange}
                  className="hidden"
                  id="patch-upload"
                />
                <span className="text-sm text-gray-600">
                  {patchFiles.length > 0
                    ? `${patchFiles.length} file(s) selected`
                    : 'Click or tap to select patch images'}
                </span>
              </div>
              {patchUploading && (
                <UploadProgress progress={patchProgress} fileName={patchFiles.map(f => f.name).join(', ')} />
              )}
              {patchError && (
                <div className="text-red-600 text-sm">{patchError}</div>
              )}
              <Button
                onClick={uploadPatches}
                disabled={patchUploading || !patchFiles.length}
                className="w-full"
              >
                {patchUploading ? 'Analyzing...' : 'Analyze Patches'}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Slides List */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <FileText className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900">Analysis History</h2>
                    <p className="text-sm text-gray-600">
                      {slides.length} slide{slides.length !== 1 ? 's' : ''} analyzed
                    </p>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8">
                  <RefreshCw className="h-8 w-8 text-gray-400 animate-spin mx-auto mb-2" />
                  <p className="text-gray-500">Loading slides...</p>
                </div>
              ) : slides.length === 0 ? (
                <div className="text-center py-8">
                  <Microscope className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No slides uploaded</h3>
                  <p className="text-gray-500">Upload your first slide to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {slides.map((slide) => (
                    <div 
                      key={slide.id} 
                      className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors cursor-pointer"
                      onClick={() => openDetailModal(slide)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 flex-1">
                          <div className="p-2 bg-gray-100 rounded-lg">
                            <ImageIcon className="h-4 w-4 text-gray-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h3 className="text-sm font-medium text-gray-900 truncate">
                              {getFileName(slide.slide_file)}
                            </h3>
                            <p className="text-xs text-gray-500">
                              Uploaded {formatDate(slide.created)}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(slide.status)}
                            <span className={getStatusBadge(slide.status)}>
                              {slide.status}
                            </span>
                          </div>
                          
                          <div className="flex items-center space-x-1">
                            {slide.status === 'COMPLETED' && slide.overview_map_url && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  window.open(slide.overview_map_url, '_blank');
                                }}
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            )}
                            
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteSlide(slide.id);
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                      
                      {slide.summary && (
                        <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-2 mb-1">
                            <BarChart3 className="h-4 w-4 text-gray-600" />
                            <span className="text-sm font-medium text-gray-900">Analysis Result</span>
                          </div>
                          <p className="text-sm text-gray-700">{slide.summary}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Patch Results */}
      {patchResults && patchResults.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <h2 className="text-xl font-semibold text-gray-900">Patch Analysis Results</h2>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {patchResults.map((result, idx) => (
                <div key={idx} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-2">
                    <img src={result.image_url} alt="patch" className="w-12 h-12 object-cover rounded" />
                    <div>
                      <div className="text-sm font-medium text-gray-900">{result.class_name}</div>
                      <div className="text-xs text-gray-600">Prob: {result.probabilities?.join(', ')}</div>
                    </div>
                  </div>
                  <div className="flex space-x-2 mt-2">
                    {result.gradcam_url && (
                      <img src={result.gradcam_url} alt="GradCAM" className="w-20 h-20 object-cover rounded border" />
                    )}
                    {result.saliency_url && (
                      <img src={result.saliency_url} alt="Saliency" className="w-20 h-20 object-cover rounded border" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detail Modal */}
      <SlideDetailModal
        slide={selectedSlide}
        isOpen={detailModalOpen}
        onClose={() => setDetailModalOpen(false)}
      />
    </div>
  );
}
