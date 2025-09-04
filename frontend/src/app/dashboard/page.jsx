"use client"

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import axios from 'axios';
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  FileText, 
  Video, 
  Dna, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Activity,
  Users,
  Database,
  Zap,
  BarChart,
  LineChart,
  RefreshCw
} from 'lucide-react';
import { 
  LineChart as RechartsLineChart,
  Line,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Label,
  RadialBarChart,
  RadialBar
} from 'recharts';
import { Card, CardHeader, CardContent } from '@/components/ui/card';

export default function DashboardPage() {
  const { token } = useAuth();
  const router = useRouter();
  const [analytics, setAnalytics] = useState({
    slides: { total: 0, completed: 0, pending: 0, failed: 0 },
    videos: { total: 0, completed: 0, pending: 0, failed: 0 },
    genomic: { total: 0, completed: 0, pending: 0, failed: 0 },
    recentActivity: [],
    performance: { avgProcessingTime: 0, successRate: 0 },
    modelStats: { wsiAccuracy: 0, polypDetection: 0, genomicAccuracy: 0 }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Smart API base URL detection for Docker vs local development
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Chart colors - updated to match design system
  const COLORS = ['#005EB8', '#76C043', '#FFB81C', '#B00020', '#0288D1'];



  useEffect(() => {
    console.log('Dashboard useEffect - token:', token); // Debug log
    if (token) {
      fetchAnalytics();
    } else {
      console.log('No token found, redirecting to login');
      router.push('/login');
    }
  }, [token, router]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      
      // Debug current configuration
      console.log('🔍 DEBUG: API_BASE =', API_BASE);
      console.log('🔍 DEBUG: process.env.NEXT_PUBLIC_API_URL =', process.env.NEXT_PUBLIC_API_URL);
      console.log('🔍 DEBUG: window.location =', window.location.href);
      
      // Test basic connectivity first
      console.log('🔗 Testing basic connectivity...');
      try {
        const basicTest = await fetch(`${API_BASE}/api/health/`);
        console.log('✅ Health check response:', basicTest.status, basicTest.statusText);
      } catch (connectError) {
        console.error('❌ Basic connectivity failed:', connectError.message);
        setError(`Cannot connect to backend at ${API_BASE}. Is the backend running?`);
        setLoading(false);
        return;
      }
      
      const headers = { Authorization: `Token ${token}` };
      console.log('📡 Attempting analytics request to:', `${API_BASE}/api/analytics/`);
      const analyticsRes = await axios.get(`${API_BASE}/api/analytics`, { headers });
      const data = analyticsRes.data;

      setAnalytics({
        slides: data.slides || { total: 0, completed: 0, pending: 0, failed: 0 },
        videos: data.videos || { total: 0, completed: 0, pending: 0, failed: 0 },
        genomic: data.genomic || { total: 0, completed: 0, pending: 0, failed: 0 },
        recentActivity: data.recent_activity?.map(item => ({
          type: item.type,
          title: item.title,
          status: item.status,
          timestamp: item.timestamp,
          icon: item.type === 'slide' ? FileText : item.type === 'video' ? Video : Dna,
          summary: item.summary
        })) || [],
        performance: {
          avgProcessingTime: data.performance?.avgProcessingTime || 0,
          successRate: data.performance?.successRate || 0
        },
        modelStats: {
          wsiAccuracy: data.model_stats?.wsi_accuracy || 0,
          polypDetection: data.model_stats?.polyp_detection || 0,
          genomicAccuracy: data.model_stats?.genomic_accuracy || 0
        }
      });
    } catch (err) {
      setError('Failed to fetch analytics data');
      console.error('Analytics error:', err);
      console.error('Error details:', {
        message: err.message,
        code: err.code,
        config: err.config?.url,
        response: err.response?.status,
        responseData: err.response?.data
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'COMPLETED': return <CheckCircle className="h-4 w-4 text-[#007F3B]" />;
      case 'FAILED': return <XCircle className="h-4 w-4 text-[#B00020]" />;
      case 'PENDING': return <Clock className="h-4 w-4 text-[#FFB81C]" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'COMPLETED': return 'text-[#007F3B]';
      case 'FAILED': return 'text-[#B00020]';
      case 'PENDING': return 'text-[#FFB81C]';
      default: return 'text-gray-600';
    }
  };

  // Chart data preparation
  const analysisTypeData = [
    { name: 'WSI Analysis', value: analytics.slides?.total || 0, color: '#005EB8' },
    { name: 'Polyp Detection', value: analytics.videos?.total || 0, color: '#76C043' },
    { name: 'Genomic Analysis', value: analytics.genomic?.total || 0, color: '#FFB81C' }
  ];

  const statusData = [
    { name: 'Completed', value: (analytics.slides?.completed || 0) + (analytics.videos?.completed || 0) + (analytics.genomic?.completed || 0), color: '#007F3B' },
    { name: 'Pending', value: (analytics.slides?.pending || 0) + (analytics.videos?.pending || 0) + (analytics.genomic?.pending || 0), color: '#FFB81C' },
    { name: 'Failed', value: (analytics.slides?.failed || 0) + (analytics.videos?.failed || 0) + (analytics.genomic?.failed || 0), color: '#B00020' }
  ];

  const modelAccuracyData = [
    { name: 'WSI', accuracy: analytics.modelStats?.wsiAccuracy || 0, color: '#005EB8' },
    { name: 'Polyp', accuracy: analytics.modelStats?.polypDetection || 0, color: '#76C043' },
    { name: 'Genomic', accuracy: analytics.modelStats?.genomicAccuracy || 0, color: '#FFB81C' }
  ];

  // Mock time series data for processing time trend
  const processingTimeData = [
    { date: '2024-01', time: 3.2 },
    { date: '2024-02', time: 2.8 },
    { date: '2024-03', time: 2.5 },
    { date: '2024-04', time: 2.1 },
    { date: '2024-05', time: 1.9 },
    { date: '2024-06', time: 1.7 }
  ];

  // Redirect to login if not authenticated
  if (!token) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#005EB8]"></div>
        <p className="ml-3">Redirecting to login...</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#005EB8]"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-[#B00020]">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Analytics and insights for your medical analysis platform</p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={fetchAnalytics}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 bg-[#005EB8] text-white rounded-lg shadow-md hover:bg-[#004a94] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <Activity className="h-4 w-4" />
            <span>Last updated: {new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="shadow-lg rounded-xl">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Analyses</p>
                <p className="text-2xl font-bold text-gray-900">
                  {(analytics.slides?.total || 0) + (analytics.videos?.total || 0) + (analytics.genomic?.total || 0)}
                </p>
              </div>
              <Database className="h-8 w-8 text-[#005EB8]" />
            </div>
          </CardContent>
        </Card>
        <Card className="shadow-lg rounded-xl">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-gray-900">
                  {(analytics.performance?.successRate || 0).toFixed(1)}%
                </p>
              </div>
              <CheckCircle className="h-8 w-8 text-[#007F3B]" />
            </div>
          </CardContent>
        </Card>
        <Card className="shadow-lg rounded-xl">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Processing Time</p>
                <p className="text-2xl font-bold text-gray-900">
                  {(analytics.performance?.avgProcessingTime || 0).toFixed(1)}s
                </p>
              </div>
              <Zap className="h-8 w-8 text-[#FFB81C]" />
            </div>
          </CardContent>
        </Card>
        <Card className="shadow-lg rounded-xl">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Users</p>
                <p className="text-2xl font-bold text-gray-900">1</p>
              </div>
              <Users className="h-8 w-8 text-[#0288D1]" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Analysis Type Distribution */}
        <Card className="shadow-xl rounded-2xl bg-white">
          <CardHeader>
            <h3 className="text-lg font-bold text-neutral-800">Analysis Type Distribution</h3>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RechartsPieChart>
                <Pie
                  data={analysisTypeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {analysisTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </RechartsPieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Status Distribution */}
        <Card className="shadow-xl rounded-2xl bg-white">
          <CardHeader>
            <h3 className="text-lg font-bold text-neutral-800">Analysis Status</h3>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RechartsBarChart data={statusData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#005EB8" />
              </RechartsBarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Model Performance */}
      <Card className="shadow-xl rounded-2xl bg-white">
        <CardHeader>
          <h3 className="text-lg font-bold text-neutral-800">Model Performance</h3>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {modelAccuracyData.map((model, index) => (
              <div key={index} className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <div 
                    className="w-20 h-20 rounded-full flex items-center justify-center text-white font-bold shadow-lg"
                  >
                    {(model.accuracy || 0).toFixed(1)}%
                  </div>
                </div>
                <h4 className="font-semibold text-gray-900">{model.name} Accuracy</h4>
                <p className="text-sm text-gray-600">Model Performance</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card className="shadow-xl rounded-2xl bg-white">
        <CardHeader>
          <h3 className="text-lg font-bold text-neutral-800">Recent Activity</h3>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {(analytics.recentActivity || []).map((activity, index) => (
              <div key={index} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg shadow-sm">
                <div className="flex-shrink-0">
                  <activity.icon className="h-6 w-6 text-[#005EB8]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                  <p className="text-sm text-gray-500">{activity.summary}</p>
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusIcon(activity.status)}
                  <span className={`text-sm font-medium ${getStatusColor(activity.status)}`}>
                    {activity.status}
                  </span>
                </div>
                <div className="text-sm text-gray-500">
                  {new Date(activity.timestamp).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
