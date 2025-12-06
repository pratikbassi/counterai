import { useState, useRef } from 'react';
import { uploadFile, type UploadResponse } from '../services/api';
import './FileUpload.css';

export default function FileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const response = await uploadFile(file);
      setResult(response);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during upload');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="file-upload-container">
      <h1>File Upload</h1>
      <p className="description">Upload a file to generate its SHA-256 hash</p>
      
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-wrapper">
          <input
            ref={fileInputRef}
            type="file"
            id="file-input"
            onChange={handleFileChange}
            disabled={uploading}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            {file ? file.name : 'Choose a file'}
          </label>
        </div>

        {file && (
          <div className="file-info">
            <p><strong>Selected:</strong> {file.name}</p>
            <p><strong>Size:</strong> {formatFileSize(file.size)}</p>
            <p><strong>Type:</strong> {file.type || 'Unknown'}</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className="success-message">
            <h3>Upload Successful!</h3>
            <div className="result-details">
              <p><strong>Hash:</strong> <code>{result.hash}</code></p>
              <p><strong>Filename:</strong> {result.filename}</p>
              <p><strong>Size:</strong> {formatFileSize(result.size)}</p>
              <p><strong>Saved at:</strong> {result.saved_at}</p>
            </div>
          </div>
        )}

        <div className="button-group">
          <button
            type="submit"
            disabled={!file || uploading}
            className="upload-button"
          >
            {uploading ? 'Uploading...' : 'Upload File'}
          </button>
          
          {(file || result) && (
            <button
              type="button"
              onClick={handleReset}
              disabled={uploading}
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

