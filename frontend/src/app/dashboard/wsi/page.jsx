import React, { useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from '@/contexts/AuthContext';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';

export default function WSIPage() {
  const { token } = useContext(AuthContext);
  const [wsiFile, setWsiFile] = useState(null);
  const [patchFiles, setPatchFiles] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [patchResult, setPatchResult] = useState(null);
  const [patchOverlays, setPatchOverlays] = useState({ grad: [], smooth: [] });
  const [uploadingWsi, setUploadingWsi] = useState(false);
  const [uploadingPatch, setUploadingPatch] = useState(false);

  useEffect(() => {
    if (token) fetchJobs();
  }, [token]);

  const fetchJobs = async () => {
    try {
      const res = await axios.get('/api/slides/', {
        headers: { Authorization: `Token ${token}` }
      });
      setJobs(res.data);
    } catch (err) {
      console.error('Failed to fetch WSI jobs', err);
    }
  };

  const handleWsiChange = e => setWsiFile(e.target.files[0]);
  const handlePatchChange = e => setPatchFiles(Array.from(e.target.files));

  const uploadWsi = async () => {
    if (!wsiFile) return;
    setUploadingWsi(true);
    const form = new FormData();
    form.append('slide', wsiFile);
    try {
      await axios.post('/api/slides/', form, {
        headers: {
          Authorization: `Token ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      fetchJobs();
    } catch (err) {
      console.error('WSI upload failed', err);
    } finally {
      setUploadingWsi(false);
      setWsiFile(null);
    }
  };

  const uploadPatch = async () => {
    if (!patchFiles.length) return;
    setUploadingPatch(true);
    const form = new FormData();
    patchFiles.forEach(file => form.append('images', file));
    try {
      const res = await axios.post('/api/patches/', form, {
        headers: {
          Authorization: `Token ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      // expecting { predictions: [...], grad_overlays: [url,...], smooth_overlays: [url,...] }
      setPatchResult(res.data.predictions);
      setPatchOverlays({ grad: res.data.grad_overlays, smooth: res.data.smooth_overlays });
    } catch (err) {
      console.error('Patch upload failed', err);
    } finally {
      setUploadingPatch(false);
      setPatchFiles([]);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">WSI & Patch-Level Analysis</h1>

      {/* WSI Upload Section */}
      <Card className="mb-6">
        <CardContent className="flex flex-col items-start gap-4">
          <h2 className="text-xl font-semibold">Whole-Slide Image</h2>
          <Input type="file" accept=".svs,.tiff" onChange={handleWsiChange} />
          <Button onClick={uploadWsi} disabled={uploadingWsi || !wsiFile}>
            {uploadingWsi ? 'Uploading WSI...' : 'Upload WSI'}
          </Button>
        </CardContent>
      </Card>

      {/* Patch-Level Upload Section */}
      <Card className="mb-6">
        <CardContent className="flex flex-col items-start gap-4">
          <h2 className="text-xl font-semibold">Patch-Level Images</h2>
          <Input type="file" accept="image/*" multiple onChange={handlePatchChange} />
          <Button onClick={uploadPatch} disabled={uploadingPatch || !patchFiles.length}>
            {uploadingPatch ? 'Uploading Patches...' : 'Upload Patches'}
          </Button>
        </CardContent>
      </Card>

      {/* Patch Results */}
      {patchResult && (
        <Card className="mb-6">
          <CardContent>
            <h2 className="text-xl font-semibold mb-2">Patch Predictions</h2>
            <ul className="list-disc pl-5">
              {patchResult.map((pred, i) => (
                <li key={i}>{pred}</li>
              ))}
            </ul>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div>
                <h3 className="font-medium mb-1">Grad-CAM</h3>
                <div className="space-y-2">
                  {patchOverlays.grad.map((url, idx) => (
                    <img key={idx} src={url} alt={`grad-cam-${idx}`} className="border rounded" />
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-medium mb-1">SmoothGrad</h3>
                <div className="space-y-2">
                  {patchOverlays.smooth.map((url, idx) => (
                    <img key={idx} src={url} alt={`smooth-grad-${idx}`} className="border rounded" />
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* WSI Jobs List */}
      <h2 className="text-xl font-semibold mb-2">WSI Analysis Jobs</h2>
      <div className="space-y-4">
        {jobs.map(job => (
          <Card key={job.id} className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium">Job #{job.id}</p>
                <p>Status: {job.status}</p>
              </div>
              {(job.status === 'PENDING' || job.status === 'RUNNING') && (
                <Progress value={job.progress * 100} className="w-32" />
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
