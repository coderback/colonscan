import React from 'react';

export function Input({ 
  className = '', 
  type = 'text',
  error = false,
  ...props 
}) {
  const baseClasses = 'block w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-500 focus:border-[#005EB8] focus:outline-none focus:ring-1 focus:ring-[#005EB8] disabled:bg-gray-50 disabled:text-gray-500 transition-colors duration-200';
  const errorClasses = error ? 'border-[#B00020] focus:border-[#B00020] focus:ring-[#B00020]' : '';
  
  return (
    <input
      type={type}
      className={`${baseClasses} ${errorClasses} ${className}`}
      {...props}
    />
  );
} 