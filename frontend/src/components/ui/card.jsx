import React from 'react';

export function Card({ children, className = '', ...props }) {
  return (
    <div 
      className={`bg-white border border-neutral-200 rounded-lg shadow-lg ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardContent({ children, className = '', ...props }) {
  return (
    <div className={`px-6 py-4 text-sm text-neutral-700 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '', ...props }) {
  return (
    <div className={`px-6 py-4 border-b border-gray-200 text-lg font-bold text-neutral-800 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({ children, className = '', ...props }) {
  return (
    <div className={`px-6 py-4 border-t border-gray-200 text-xs text-neutral-600 ${className}`} {...props}>
      {children}
    </div>
  );
} 