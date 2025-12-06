const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';

export interface UploadResponse {
  hash: string;
  filename: string;
  size: number;
  saved_at: string;
}

export interface ApiError {
  error: string;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/file_hashes/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error: ApiError = await response.json();
    throw new Error(error.error || `Upload failed: ${response.statusText}`);
  }

  return response.json();
}

