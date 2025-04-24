// frontend/src/App.js
import React, { useEffect, useState } from 'react';
import { Container, Typography, Card, CardContent, Button } from '@mui/material';

function App() {
  const [health, setHealth] = useState('Loading...');

  useEffect(() => {
    // Replace with your backend API URL
    fetch('/api/health/')
      .then((res) => res.json())
      .then((data) => setHealth(data.status))
      .catch((err) => setHealth('Error fetching status'));
  }, []);

  return (
    <Container maxWidth="sm" style={{ marginTop: '2rem' }}>
      <Card>
        <CardContent>
          <Typography variant="h4" component="div">
            Welcome to ColonoScan
          </Typography>
          <Typography variant="body1" color="text.secondary" style={{ marginTop: '1rem' }}>
            {health}
          </Typography>
          <Button variant="contained" color="primary" style={{ marginTop: '1rem' }}>
            Get Started
          </Button>
        </CardContent>
      </Card>
    </Container>
  );
}

export default App;
