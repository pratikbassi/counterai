import { useEffect, useRef, useState } from 'react';
import { fetchFileHashStatus, uploadFile, type UploadResponse } from '../services/api';
import './FileUpload.css';

const POLL_INTERVAL_MS = 2000;
const POLL_ATTEMPTS = 30;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = window.setTimeout(() => resolve(), ms);
    if (signal?.aborted) {
      window.clearTimeout(t);
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const onAbort = () => {
      window.clearTimeout(t);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export default function FileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [polling, setPolling] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollAbortRef = useRef<AbortController | null>(null);

  useEffect(() => () => pollAbortRef.current?.abort(), []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file size (25MB)
      const maxSize = 25 * 1024 * 1024; // 25MB in bytes
      if (selectedFile.size > maxSize) {
        setError('File size exceeds maximum allowed size of 25MB');
        setFile(null);
        return;
      }
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please choose an image file (JPEG, PNG, WebP, or GIF)');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError('Please select an image to test');
      return;
    }

    pollAbortRef.current?.abort();
    const ac = new AbortController();
    pollAbortRef.current = ac;
    const signal = ac.signal;

    setUploading(true);
    setPolling(false);
    setError(null);
    setResult(null);

    try {
      const response = await uploadFile(file);
      setResult(response);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      if (response.ai_status !== 'unknown') {
        return;
      }

      setPolling(true);
      try {
        for (let i = 0; i < POLL_ATTEMPTS; i++) {
          await sleep(POLL_INTERVAL_MS, signal);

          const status = await fetchFileHashStatus(response.hash, signal);
          setResult((prev) =>
            prev
              ? {
                  ...prev,
                  found_in_database: status.found_in_database,
                  ai_status: status.ai_status,
                }
              : prev
          );

          if (status.ai_status !== 'unknown') {
            return;
          }
        }
        setError('Detection is still running. Refresh and try again in a minute.');
      } finally {
        setPolling(false);
      }
    } catch (err) {
      const message =
        err instanceof Error && err.name === 'AbortError'
          ? null
          : err instanceof Error
            ? err.message
            : 'An error occurred while testing the file';
      if (message) {
        setError(message);
      }
    } finally {
      setUploading(false);
      setPolling(false);
    }
  };

  const handleReset = () => {
    pollAbortRef.current?.abort();
    setFile(null);
    setResult(null);
    setError(null);
    setPolling(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const showProcessing = polling || uploading;

  return (
    <div className="file-upload-container">
      <h1>File tester</h1>
      <p className="description">Test your images for AI.</p>

      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-wrapper">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            id="file-input"
            onChange={handleFileChange}
            disabled={uploading || polling}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            {file ? file.name : 'Choose an image'}
          </label>
        </div>

        {file && (
          <div className="file-info">
            <p><strong>Selected:</strong> {file.name}</p>
            <p><strong>Size:</strong> {formatFileSize(file.size)}</p>
            <p><strong>Type:</strong> {file.type || 'Unknown'}</p>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {showProcessing && (
          <div className="file-info">
            <p><strong>{uploading ? 'Uploading…' : 'Processing detection…'}</strong></p>
          </div>
        )}

        {result && !showProcessing && (
          <div className="success-message">
            <h3>
              {result.found_in_database
                ? 'Image found in database'
                : 'Image not in database — added'}
            </h3>
            <div className="result-details">
              <p>
                <strong>AI content:</strong>{' '}
                {result.ai_status === 'ai_detected'
                  ? 'AI Detected'
                  : result.ai_status === 'ai_not_detected'
                    ? 'AI Not Detected'
                    : 'Unknown AI content'}
              </p>
              <p>
                <strong>Hash:</strong> <code>{result.hash}</code>
              </p>
              <p><strong>Filename:</strong> {result.filename}</p>
              <p><strong>Size:</strong> {formatFileSize(result.size)}</p>
              <p><strong>Saved at:</strong> {result.saved_at}</p>
            </div>
          </div>
        )}

        <div className="button-group">
          <button
            type="submit"
            disabled={!file || uploading || polling}
            className="upload-button"
          >
            {uploading || polling ? 'Testing…' : 'Test image'}
          </button>

          {(file || result) && (
            <button
              type="button"
              onClick={handleReset}
              disabled={uploading || polling}
              className="reset-button"
            >
              Reset
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
