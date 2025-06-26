import React from 'react';
import { X, Download, Eye, BarChart3, Calendar, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function SlideDetailModal({ slide, isOpen, onClose }) {
  if (!isOpen || !slide) return null;

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getFileName = (filePath) => {
    return filePath.split('/').pop();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'PENDING':
        return 'text-yellow-600 bg-yellow-50';
      case 'RUNNING':
        return 'text-blue-600 bg-blue-50';
      case 'COMPLETED':
        return 'text-green-600 bg-green-50';
      case 'FAILED':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 p-4 pointer-events-auto">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-gray-200">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <FileText className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Slide Details
              </h2>
              <p className="text-sm text-gray-600">
                {getFileName(slide.slide_file)}
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* File Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900">File Information</h3>
              
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <FileText className="h-4 w-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">Filename</p>
                    <p className="text-sm text-gray-600">{getFileName(slide.slide_file)}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <Calendar className="h-4 w-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">Upload Date</p>
                    <p className="text-sm text-gray-600">{formatDate(slide.created)}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <BarChart3 className="h-4 w-4 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">Analysis Status</p>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(slide.status)}`}>
                      {slide.status}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Results */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900">Analysis Results</h3>
              
              {slide.summary ? (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <BarChart3 className="h-4 w-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-900">Summary</span>
                  </div>
                  <p className="text-sm text-gray-700">{slide.summary}</p>
                </div>
              ) : (
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <p className="text-sm text-gray-500">
                    {slide.status === 'PENDING' && 'Analysis pending...'}
                    {slide.status === 'RUNNING' && 'Analysis in progress...'}
                    {slide.status === 'FAILED' && 'Analysis failed'}
                    {slide.status === 'COMPLETED' && 'No summary available'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Overview Map */}
          {slide.overview_map_url && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">Overview Heatmap</h3>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.open(slide.overview_map_url, '_blank')}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    View Full Size
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const link = document.createElement('a');
                      link.href = slide.overview_map_url;
                      link.download = `${getFileName(slide.slide_file)}_heatmap.png`;
                      link.click();
                    }}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
              
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <img
                  src={slide.overview_map_url}
                  alt="Overview heatmap"
                  className="w-full h-64 object-cover"
                />
              </div>
              
              <div className="text-sm text-gray-600">
                <p>This heatmap shows the probability distribution across the slide, with warmer colors indicating higher probability of malignancy.</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-200">
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
            <Button
              onClick={() => {
                const link = document.createElement('a');
                link.href = slide.slide_file;
                link.download = getFileName(slide.slide_file);
                link.click();
              }}
            >
              <Download className="h-4 w-4 mr-2" />
              Download Original
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
} 