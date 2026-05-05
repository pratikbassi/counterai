const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000';

/** Backend ai_status values */
export type AiStatus = 'unknown' | 'ai_detected' | 'ai_not_detected';

/** Lowercase SHA-256 hex (64 chars) */
const SHA256_HEX = /^[a-f0-9]{64}$/;

export interface UploadResponse {
  hash: string;
  filename: string;
  size: number;
  saved_at: string;
  found_in_database: boolean;
  ai_status: AiStatus;
}

export interface FileHashStatusResponse {
  hash: string;
  found_in_database: boolean;
  ai_status: AiStatus;
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

export async function fetchFileHashStatus(
  hash: string,
  signal?: AbortSignal
): Promise<FileHashStatusResponse> {
  const h = hash.trim().toLowerCase();
  if (!SHA256_HEX.test(h)) {
    throw new Error('Invalid file hash');
  }

  const response = await fetch(`${API_BASE_URL}/file_hashes/${encodeURIComponent(h)}`, {
    method: 'GET',
    signal,
  });

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Status request failed: ${response.statusText}`);
  }

  if (!response.ok && response.status !== 404) {
    const err = body as ApiError;
    throw new Error(err.error || `Status request failed: ${response.statusText}`);
  }

  return body as FileHashStatusResponse;
}
