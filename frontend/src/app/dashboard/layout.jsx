"use client"

import React, { useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { 
  Microscope, 
  Video, 
  Dna, 
  FileText, 
  LogOut, 
  User,
  Activity
} from 'lucide-react';

export default function DashboardLayout({ children }) {
  const { token, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  // Redirect if not logged in
  useEffect(() => {
    if (!token) {
      router.replace('/login');
    }
  }, [token, router]);

  // Show loading or nothing while checking authentication
  if (!token) {
    return null;
  }

  const navigation = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Activity,
      description: 'Analytics & Overview'
    },
    {
      name: 'WSI Analysis',
      href: '/dashboard/wsi',
      icon: Microscope,
      description: 'Histopathology slides'
    },
    {
      name: 'Polyp Detection',
      href: '/dashboard/polyp',
      icon: Video,
      description: 'Colonoscopy videos'
    },
    {
      name: 'Genomic Analysis',
      href: '/dashboard/genomic',
      icon: Dna,
      description: 'DNA sequencing'
    }
  ];

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0 flex items-center space-x-2">
                <Image src="/logo.png" alt="ColonoScan Logo" width={32} height={32} />
                <h1 className="text-2xl font-bold text-blue-600">ColonoScan</h1>
              </div>
              <div className="ml-8">
                <p className="text-sm text-gray-600">AI-Powered Medical Analysis Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-600">
                <User className="h-5 w-5" />
                <span className="text-sm font-medium">Admin</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <LogOut className="h-5 w-5" />
                <span className="text-sm">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
          <nav className="mt-8">
            <div className="px-4">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                Analysis Tools
              </h2>
            </div>
            <ul className="space-y-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={`flex items-center px-4 py-3 text-sm font-medium transition-colors duration-200 ${
                        isActive
                          ? 'bg-blue-50 border-r-2 border-blue-600 text-blue-700'
                          : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                      }`}
                    >
                      <item.icon className="h-5 w-5 mr-3" />
                      <div>
                        <div>{item.name}</div>
                        <div className="text-xs text-gray-500 mt-0.5">{item.description}</div>
                      </div>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-8">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
