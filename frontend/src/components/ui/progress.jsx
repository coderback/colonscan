import React from 'react';

export function Progress({ 
  value = 0, 
  max = 100,
  className = '',
  ...props 
}) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  return (
    <div 
      className={`medical-progress ${className}`}
      {...props}
    >
      <div 
        className="medical-progress-bar"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
} 