"use client"

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Microscope, AlertCircle } from 'lucide-react';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { token, login } = useAuth();
  const router = useRouter();

  // Redirect if already logged in
  useEffect(() => {
    if (token) {
      router.replace('/dashboard');
    }
  }, [token, router]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(username, password);
      router.push('/dashboard');
    } catch (err) {
      setError('Invalid credentials. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Show loading if checking authentication
  if (token) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-[#005EB8] rounded-2xl mb-4">
            <Image src="/logo.png" alt="ColonScan Logo" width={40} height={40} />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">ColonScan</h1>
          <p className="text-gray-600">AI-powered colonoscopy assistant for early detection of colorectal cancer</p>
        </div>

        {/* Login Card */}
        <Card className="shadow-xl">
          <CardHeader className="text-center pb-4">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <h2 className="text-xl font-semibold text-gray-900">Login</h2>
            </div>
            </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="flex items-center space-x-2 p-3 bg-[#B00020]/5 border border-[#B00020]/20 rounded-lg">
                  <AlertCircle className="h-4 w-4 text-[#B00020]" />
                  <span className="text-sm text-[#B00020]">{error}</span>
                </div>
              )}
              
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                  Username
                </label>
                <Input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  required
                  className="w-full"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  required
                  className="w-full"
                />
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full"
              >
                {loading ? 'Signing In...' : 'Sign In'}
              </Button>
            </form>

            {/* Sign Up Link */}
            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                Don&apos;t have an account?{' '}
                <a href="/signup" className="text-[#005EB8] hover:text-[#004a94] font-medium">
                  Sign up here
                </a>
              </p>
            </div>

            {/* Demo Credentials */}
            <div className="mt-6 p-4 bg-[#005EB8]/5 rounded-lg">
              <h3 className="text-sm font-medium text-[#005EB8] mb-2">Demo Credentials</h3>
              <div className="text-xs text-[#005EB8] space-y-1">
                <p><strong>Username:</strong> admin</p>
                <p><strong>Password:</strong> colonscan</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-sm text-gray-500">
            Secure medical colonoscopy analysis platform
          </p>
        </div>
      </div>
    </div>
  );
}
