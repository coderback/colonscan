import React from 'react';

export function Card({ children, className = '', ...props }) {
  return (
    <div 
      className={`medical-card ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardContent({ children, className = '', ...props }) {
  return (
    <div className={`medical-card-content ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '', ...props }) {
  return (
    <div className={`medical-card-header ${className}`} {...props}>
      {children}
    </div>
  );
} 