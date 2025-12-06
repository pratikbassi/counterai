require "test_helper"
require "digest"

class FileHashesControllerTest < ActionDispatch::IntegrationTest
  setup do
    @existing_hash = "abc123def456"
    @non_existing_hash = "xyz789ghi012"
    
    # Create a file hash in the database
    FileHash.create!(hash_value: @existing_hash)
  end

  test "should return true for existing hash" do
    post file_hashes_check_path, params: { hashes: [@existing_hash] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal true, json_response[@existing_hash]
  end

  test "should return false for non-existing hash" do
    post file_hashes_check_path, params: { hashes: [@non_existing_hash] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal false, json_response[@non_existing_hash]
  end

  test "should handle multiple hashes" do
    post file_hashes_check_path, params: { hashes: [@existing_hash, @non_existing_hash] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal true, json_response[@existing_hash]
    assert_equal false, json_response[@non_existing_hash]
  end

  test "should handle empty array" do
    post file_hashes_check_path, params: { hashes: [] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal({}, json_response)
  end

  test "should handle missing hashes parameter" do
    post file_hashes_check_path, params: {}, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal({}, json_response)
  end

  test "should reject non-array hashes parameter" do
    post file_hashes_check_path, params: { hashes: "not-an-array" }, as: :json
    
    assert_response :bad_request
    json_response = JSON.parse(response.body)
    assert_equal "hashes must be an array", json_response["error"]
  end

  test "should handle duplicate hashes" do
    post file_hashes_check_path, params: { hashes: [@existing_hash, @existing_hash] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal true, json_response[@existing_hash]
    assert_equal 1, json_response.keys.length
  end

  test "should ignore nil and empty string hashes" do
    post file_hashes_check_path, params: { hashes: [@existing_hash, nil, "", "   "] }, as: :json
    
    assert_response :success
    json_response = JSON.parse(response.body)
    assert_equal true, json_response[@existing_hash]
    assert_equal 1, json_response.keys.length
  end

  test "should reject requests with more than 1000 hashes" do
    large_array = 1001.times.map { |i| "hash_#{i}" }
    post file_hashes_check_path, params: { hashes: large_array }, as: :json
    
    assert_response :bad_request
    json_response = JSON.parse(response.body)
    assert_equal "maximum 1000 hashes allowed per request", json_response["error"]
  end

  test "should accept exactly 1000 hashes" do
    large_array = 1000.times.map { |i| "hash_#{i}" }
    post file_hashes_check_path, params: { hashes: large_array }, as: :json
    
    assert_response :success
  end

  # Upload endpoint tests
  test "should upload file and return hash" do
    file = fixture_file_upload("test/fixtures/files/test.txt", "text/plain")
    
    post file_hashes_upload_path, params: { file: file }
    
    assert_response :created
    json_response = JSON.parse(response.body)
    assert json_response["hash"].present?
    assert_equal "test.txt", json_response["filename"]
    assert json_response["size"].present?
    assert json_response["saved_at"].present?
    
    # Verify hash was saved to database
    assert FileHash.exists?(hash_value: json_response["hash"])
    
    # Verify file was saved to storage
    assert File.exist?(Rails.root.join(json_response["saved_at"]))
  end

  test "should reject upload without file parameter" do
    post file_hashes_upload_path, params: {}
    
    assert_response :bad_request
    json_response = JSON.parse(response.body)
    assert_equal "file parameter is required", json_response["error"]
  end

  test "should reject file larger than 25MB" do
    # Create a temporary file larger than 25MB
    large_file = Tempfile.new(["large", ".txt"])
    large_file.write("x" * (25.megabytes + 1))
    large_file.rewind
    
    upload = Rack::Test::UploadedFile.new(large_file.path, "text/plain")
    
    post file_hashes_upload_path, params: { file: upload }
    
    assert_response :content_too_large
    json_response = JSON.parse(response.body)
    assert_equal "file size exceeds maximum allowed size of 25MB", json_response["error"]
    
    large_file.close
    large_file.unlink
  end

  test "should accept file exactly 25MB" do
    # Create a temporary file exactly 25MB
    large_file = Tempfile.new(["large", ".txt"])
    large_file.write("x" * 25.megabytes)
    large_file.rewind
    
    upload = Rack::Test::UploadedFile.new(large_file.path, "text/plain")
    
    post file_hashes_upload_path, params: { file: upload }
    
    assert_response :created
    json_response = JSON.parse(response.body)
    assert_equal 25.megabytes, json_response["size"]
    
    large_file.close
    large_file.unlink
  end

  test "should generate correct SHA-256 hash" do
    # Create a file with known content
    test_content = "Hello, World!"
    file = Tempfile.new(["test", ".txt"])
    file.write(test_content)
    file.rewind
    
    upload = Rack::Test::UploadedFile.new(file.path, "text/plain")
    
    post file_hashes_upload_path, params: { file: upload }
    
    assert_response :created
    json_response = JSON.parse(response.body)
    
    # Calculate expected hash
    expected_hash = Digest::SHA256.hexdigest(test_content)
    assert_equal expected_hash, json_response["hash"]
    
    file.close
    file.unlink
  end

  test "should handle duplicate file uploads" do
    test_content = "Duplicate test content"
    file1 = Tempfile.new(["test1", ".txt"])
    file1.write(test_content)
    file1.rewind
    
    file2 = Tempfile.new(["test2", ".txt"])
    file2.write(test_content)
    file2.rewind
    
    upload1 = Rack::Test::UploadedFile.new(file1.path, "text/plain")
    upload2 = Rack::Test::UploadedFile.new(file2.path, "text/plain")
    
    # First upload
    post file_hashes_upload_path, params: { file: upload1 }
    assert_response :created
    json_response1 = JSON.parse(response.body)
    hash1 = json_response1["hash"]
    
    # Second upload with same content
    post file_hashes_upload_path, params: { file: upload2 }
    assert_response :created
    json_response2 = JSON.parse(response.body)
    hash2 = json_response2["hash"]
    
    # Hashes should be the same
    assert_equal hash1, hash2
    
    # But only one record in database
    assert_equal 1, FileHash.where(hash_value: hash1).count
    
    file1.close
    file1.unlink
    file2.close
    file2.unlink
  end

  test "should save file to organized directory structure" do
    file = fixture_file_upload("test/fixtures/files/test.txt", "text/plain")
    
    post file_hashes_upload_path, params: { file: file }
    
    assert_response :created
    json_response = JSON.parse(response.body)
    saved_path = json_response["saved_at"]
    
    # Should be in storage/uploads/YYYY/MM/DD/ format
    assert_match %r{storage/uploads/\d{4}/\d{2}/\d{2}/}, saved_path
    
    # File should exist
    assert File.exist?(Rails.root.join(saved_path))
  end

  test "should handle binary files" do
    # Create a binary file (simulate image)
    binary_content = "\x89PNG\r\n\x1a\n" + ("\x00" * 100)
    file = Tempfile.new(["test", ".png"])
    file.binmode
    file.write(binary_content)
    file.rewind
    
    upload = Rack::Test::UploadedFile.new(file.path, "image/png")
    
    post file_hashes_upload_path, params: { file: upload }
    
    assert_response :created
    json_response = JSON.parse(response.body)
    assert json_response["hash"].present?
    
    # Verify file was saved correctly
    saved_file_path = Rails.root.join(json_response["saved_at"])
    assert File.exist?(saved_file_path)
    assert_equal binary_content.bytesize, File.size(saved_file_path)
    
    file.close
    file.unlink
  end
end

