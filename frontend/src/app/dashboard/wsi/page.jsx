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

  // Smart API base URL detection for Docker vs local development  
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
        return <Clock className="h-4 w-4 text-[#FFB81C]" />;
      case 'RUNNING':
        return <Play className="h-4 w-4 text-[#005EB8] animate-pulse" />;
      case 'COMPLETED':
        return <CheckCircle className="h-4 w-4 text-[#007F3B]" />;
      case 'FAILED':
        return <AlertCircle className="h-4 w-4 text-[#B00020]" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusBadge = (status) => {
    const baseClasses = "px-2.5 py-0.5 rounded-full text-xs font-medium";
    switch (status) {
      case 'PENDING':
        return `${baseClasses} bg-[#FFB81C]/10 text-[#FFB81C]`;
      case 'RUNNING':
        return `${baseClasses} bg-[#005EB8]/10 text-[#005EB8]`;
      case 'COMPLETED':
        return `${baseClasses} bg-[#007F3B]/10 text-[#007F3B]`;
      case 'FAILED':
        return `${baseClasses} bg-[#B00020]/10 text-[#B00020]`;
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

    const formData = new FormData();
    patchFiles.forEach(file => {
      formData.append('patches', file);
    });

    try {
      const progressInterval = setInterval(() => {
        setPatchProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await axios.post(`${API_BASE}/api/patches/batch/`, formData, {
        headers: {
          Authorization: `Token ${token}`
        }
      });

      setPatchProgress(100);
      setPatchResults(response.data.results);
      
      setTimeout(() => {
        setPatchProgress(0);
        setPatchFiles([]);
      }, 1000);

    } catch (error) {
      console.error('Patch upload failed:', error);
      setPatchError('Patch upload failed. Please try again.');
    } finally {
      setPatchUploading(false);
    }
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
          <h1 className="text-3xl font-bold text-gray-900">WSI Classification</h1>
          <p className="text-gray-600 mt-1">Upload and analyze whole slide images for tissue classification</p>
        </div>
        <Button onClick={fetchSlides} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Upload Section */}
      <Card>
        <CardHeader>
          <h2 className="text-lg font-bold text-neutral-800">Upload WSI File</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Input
                type="file"
                accept=".svs,.tiff,.tif"
                onChange={handleFileSelect}
                className="flex-1"
              />
              <Button
                onClick={uploadSlide}
                disabled={!selectedFile || uploading}
                className="bg-[#005EB8] hover:bg-[#004a94]"
              >
                <Upload className="h-4 w-4 mr-2" />
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>
            </div>
            
            {uploadProgress > 0 && (
              <UploadProgress progress={uploadProgress} />
            )}
            
            {selectedFile && (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <FileText className="h-4 w-4" />
                <span>Selected: {selectedFile.name}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Patch Upload Section */}
      <Card>
        <CardHeader>
          <h2 className="text-lg font-bold text-neutral-800">Batch Patch Analysis</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Input
                type="file"
                accept="image/*"
                multiple
                onChange={handlePatchChange}
                className="flex-1"
              />
              <Button
                onClick={uploadPatches}
                disabled={!patchFiles.length || patchUploading}
                variant="outline"
              >
                <Upload className="h-4 w-4 mr-2" />
                {patchUploading ? 'Processing...' : `Upload ${patchFiles.length} Patches`}
              </Button>
            </div>
            
            {patchProgress > 0 && (
              <UploadProgress progress={patchProgress} />
            )}
            
            {patchError && (
              <div className="flex items-center space-x-2 p-3 bg-[#B00020]/5 border border-[#B00020]/20 rounded-lg">
                <AlertCircle className="h-4 w-4 text-[#B00020]" />
                <span className="text-sm text-[#B00020]">{patchError}</span>
              </div>
            )}
            
            {patchResults.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-medium text-gray-900">Patch Results:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {patchResults.map((result, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-900">
                          Patch {index + 1}
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          result.classification === 'benign' 
                            ? 'bg-[#007F3B]/10 text-[#007F3B]' 
                            : 'bg-[#B00020]/10 text-[#B00020]'
                        }`}>
                          {result.classification}
                        </span>
                      </div>
                      <p className="text-xs text-gray-600">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Slides List */}
      <Card>
        <CardHeader>
          <h2 className="text-lg font-bold text-neutral-800">Uploaded Slides</h2>
        </CardHeader>
        <CardContent>
          {slides.length === 0 ? (
            <div className="text-center py-8">
              <Microscope className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No slides uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-4">
              {slides.map((slide) => (
                <div key={slide.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="flex-shrink-0">
                      <ImageIcon className="h-8 w-8 text-[#005EB8]" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">
                        {getFileName(slide.slide_file)}
                      </p>
                      <p className="text-sm text-gray-500">
                        Uploaded: {formatDate(slide.created)}
                      </p>
                      {slide.summary && (
                        <p className="text-xs text-gray-600 mt-1">{slide.summary}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(slide.status)}
                      <span className={getStatusBadge(slide.status)}>
                        {slide.status}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Button
                        onClick={() => openDetailModal(slide)}
                        variant="outline"
                        size="sm"
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      
                      {slide.status === 'COMPLETED' && (
                        <Button
                          onClick={() => window.open(slide.slide_file, '_blank')}
                          variant="outline"
                          size="sm"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}
                      
                      <Button
                        onClick={() => deleteSlide(slide.id)}
                        variant="outline"
                        size="sm"
                        className="text-[#B00020] hover:bg-[#B00020]/10"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Slide Detail Modal */}
      {selectedSlide && (
        <SlideDetailModal
          slide={selectedSlide}
          isOpen={detailModalOpen}
          onClose={() => setDetailModalOpen(false)}
        />
      )}
    </div>
  );
}
