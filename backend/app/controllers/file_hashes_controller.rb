require "digest"
require "fileutils"
require "securerandom"

class FileHashesController < ApplicationController
  # Skip CSRF token verification for API endpoint
  skip_before_action :verify_authenticity_token
  
  # Add CORS headers for cross-origin requests
  after_action :set_cors_headers

  # Maximum file size: 25MB
  MAX_FILE_SIZE = 25.megabytes

  # Handle CORS preflight requests
  def options
    head :ok
  end

  # POST /file_hashes/check
  # Accepts: { "hashes": ["hash1", "hash2", ...] }
  # Returns: { "hash1": true, "hash2": false, ... }
  def check
    hashes = params[:hashes] || []

    unless hashes.is_a?(Array)
      render json: { error: "hashes must be an array" }, status: :bad_request
      return
    end

    # Limit batch size for performance and security
    if hashes.length > 1000
      render json: { error: "maximum 1000 hashes allowed per request" }, status: :bad_request
      return
    end

    results = FileHash.exist?(hashes)
    
    render json: results, status: :ok
  end

  # POST /file_hashes/upload
  # Accepts: multipart/form-data with "file" parameter
  # Returns: { "hash": "sha256_hash", "filename": "original_filename" }
  def upload
    file = params[:file]

    unless file
      render json: { error: "file parameter is required" }, status: :bad_request
      return
    end

    # Validate file size
    if file.size > MAX_FILE_SIZE
      render json: { error: "file size exceeds maximum allowed size of 25MB" }, status: :content_too_large
      return
    end

    begin
      # Generate SHA-256 hash and save file in one pass for efficiency
      hash, file_path = generate_hash_and_save_file(file)

      # Check if hash already exists (or create it)
      _file_hash_record = FileHash.find_or_create_by!(hash_value: hash)

      # Enqueue detection job with the absolute file path
      absolute_file_path = Rails.root.join(file_path).to_s
      DetectorJob.perform_later(absolute_file_path)

      render json: {
        hash: hash,
        filename: file.original_filename,
        size: file.size,
        saved_at: file_path
      }, status: :created
    rescue ActiveRecord::RecordInvalid => e
      render json: { error: "failed to save hash: #{e.message}" }, status: :unprocessable_entity
    rescue StandardError => e
      Rails.logger.error "File upload error: #{e.message}"
      render json: { error: "failed to process file upload" }, status: :internal_server_error
    end
  end

  private

  def set_cors_headers
    # Allow requests from frontend in development
    origin = request.headers['Origin']
    allowed_origins = ['http://localhost:5173', 'http://127.0.0.1:5173']
    
    if origin && (Rails.env.development? || allowed_origins.include?(origin))
      response.headers['Access-Control-Allow-Origin'] = origin
    end
    
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Max-Age'] = '86400' # 24 hours
  end

  # Generate hash and save file in a single pass for efficiency
  def generate_hash_and_save_file(file)
    digest = Digest::SHA256.new
    
    # Reset file pointer to beginning
    file.rewind
    
    # Create storage directory structure: storage/uploads/YYYY/MM/DD/
    upload_dir = Rails.root.join("storage", "uploads", Time.current.strftime("%Y/%m/%d"))
    FileUtils.mkdir_p(upload_dir)

    # Prepare filename (we'll use a temporary name first, then rename after we have the hash)
    file_extension = File.extname(file.original_filename)
    file_basename = File.basename(file.original_filename, file_extension)
    temp_filename = "temp_#{SecureRandom.hex(8)}#{file_extension}"
    temp_file_path = upload_dir.join(temp_filename)

    # Read file in chunks, updating hash and writing to disk simultaneously
    File.open(temp_file_path, "wb") do |output_file|
      while chunk = file.read(8192)
        digest.update(chunk)
        output_file.write(chunk)
      end
    end

    # Generate final hash
    hash = digest.hexdigest

    # Rename file with hash prefix
    safe_filename = "#{hash[0..15]}_#{sanitize_filename(file_basename)}#{file_extension}"
    final_file_path = upload_dir.join(safe_filename)
    FileUtils.mv(temp_file_path, final_file_path)

    # Return hash and relative path
    [hash, final_file_path.relative_path_from(Rails.root).to_s]
  end

  def sanitize_filename(filename)
    # Remove any characters that could be problematic in filenames
    filename.gsub(/[^0-9A-Za-z.\-_]/, "_")
  end
end

