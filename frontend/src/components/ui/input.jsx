import React from 'react';

export function Input({ 
  className = '', 
  type = 'text',
  ...props 
}) {
  return (
    <input
      type={type}
      className={`medical-input ${className}`}
      {...props}
    />
  );
} 