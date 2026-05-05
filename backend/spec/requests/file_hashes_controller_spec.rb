require "rails_helper"
require "digest"
require "stringio"

RSpec.describe "FileHashesController", type: :request do
  let(:existing_hash) { "abc123def456" }
  let(:non_existing_hash) { "xyz789ghi012" }

  before do
    # Create a file hash in the database
    FileHash.create!(hash_value: existing_hash)
  end

  describe "POST /file_hashes/check" do
    it "should return true for existing hash" do
      post file_hashes_check_path, params: { hashes: [existing_hash] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response[existing_hash]).to eq(true)
    end

    it "should return false for non-existing hash" do
      post file_hashes_check_path, params: { hashes: [non_existing_hash] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response[non_existing_hash]).to eq(false)
    end

    it "should handle multiple hashes" do
      post file_hashes_check_path, params: { hashes: [existing_hash, non_existing_hash] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response[existing_hash]).to eq(true)
      expect(json_response[non_existing_hash]).to eq(false)
    end

    it "should handle empty array" do
      post file_hashes_check_path, params: { hashes: [] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response).to eq({})
    end

    it "should handle missing hashes parameter" do
      post file_hashes_check_path, params: {}, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response).to eq({})
    end

    it "should reject non-array hashes parameter" do
      post file_hashes_check_path, params: { hashes: "not-an-array" }, as: :json
      
      expect(response).to have_http_status(:bad_request)
      json_response = JSON.parse(response.body)
      expect(json_response["error"]).to eq("hashes must be an array")
    end

    it "should handle duplicate hashes" do
      post file_hashes_check_path, params: { hashes: [existing_hash, existing_hash] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response[existing_hash]).to eq(true)
      expect(json_response.keys.length).to eq(1)
    end

    it "should ignore nil and empty string hashes" do
      post file_hashes_check_path, params: { hashes: [existing_hash, nil, "", "   "] }, as: :json
      
      expect(response).to have_http_status(:success)
      json_response = JSON.parse(response.body)
      expect(json_response[existing_hash]).to eq(true)
      expect(json_response.keys.length).to eq(1)
    end

    it "should reject requests with more than 1000 hashes" do
      large_array = 1001.times.map { |i| "hash_#{i}" }
      post file_hashes_check_path, params: { hashes: large_array }, as: :json
      
      expect(response).to have_http_status(:bad_request)
      json_response = JSON.parse(response.body)
      expect(json_response["error"]).to eq("maximum 1000 hashes allowed per request")
    end

    it "should accept exactly 1000 hashes" do
      large_array = 1000.times.map { |i| "hash_#{i}" }
      post file_hashes_check_path, params: { hashes: large_array }, as: :json
      
      expect(response).to have_http_status(:success)
    end
  end

  describe "POST /file_hashes/upload" do
    def upload_minimal_png(original_filename: "minimal.png")
      Rack::Test::UploadedFile.new(
        Rails.root.join("spec/fixtures/files/minimal.png"),
        "image/png",
        original_filename:
      )
    end

    it "should upload file and return hash" do
      file = upload_minimal_png

      post file_hashes_upload_path, params: { file: file }

      expect(response).to have_http_status(:created)
      json_response = JSON.parse(response.body)
      expect(json_response["hash"]).to be_present
      expect(json_response["filename"]).to eq("minimal.png")
      expect(json_response["size"]).to be_present
      expect(json_response["saved_at"]).to be_present

      # Verify hash was saved to database
      expect(FileHash.exists?(hash_value: json_response["hash"])).to be true

      # Verify file was saved to storage
      expect(File.exist?(Rails.root.join(json_response["saved_at"]))).to be true
    end

    it "should reject non-image uploads" do
      file = fixture_file_upload("spec/fixtures/files/test.txt", "text/plain")

      post file_hashes_upload_path, params: { file: file }

      expect(response).to have_http_status(:unsupported_media_type)
      json_response = JSON.parse(response.body)
      expect(json_response["error"]).to eq("unsupported file type")
    end

    it "should reject image content_type with mismatched extension" do
      file = Rack::Test::UploadedFile.new(
        Rails.root.join("spec/fixtures/files/minimal.png"),
        "image/png",
        original_filename: "trick.txt"
      )

      post file_hashes_upload_path, params: { file: file }

      expect(response).to have_http_status(:unsupported_media_type)
    end

    it "should reject upload without file parameter" do
      post file_hashes_upload_path, params: {}
      
      expect(response).to have_http_status(:bad_request)
      json_response = JSON.parse(response.body)
      expect(json_response["error"]).to eq("file parameter is required")
    end

    it "should reject file larger than 25MB" do
      # Create a temporary file larger than 25MB
      large_file = Tempfile.new(["large", ".png"])
      large_file.write("x" * (25.megabytes + 1))
      large_file.rewind

      upload = Rack::Test::UploadedFile.new(large_file.path, "image/png")

      post file_hashes_upload_path, params: { file: upload }

      expect(response).to have_http_status(:content_too_large)
      json_response = JSON.parse(response.body)
      expect(json_response["error"]).to eq("file size exceeds maximum allowed size of 25MB")

      large_file.close
      large_file.unlink
    end

    it "should accept file exactly 25MB" do
      large_file = Tempfile.new(["large", ".png"])
      large_file.write("y" * 25.megabytes)
      large_file.rewind

      upload = Rack::Test::UploadedFile.new(large_file.path, "image/png")

      post file_hashes_upload_path, params: { file: upload }

      expect(response).to have_http_status(:created)
      json_response = JSON.parse(response.body)
      expect(json_response["size"]).to eq(25.megabytes)

      large_file.close
      large_file.unlink
    end

    it "should generate correct SHA-256 hash" do
      test_content = File.binread(Rails.root.join("spec/fixtures/files/minimal.png"))

      upload = Rack::Test::UploadedFile.new(
        StringIO.new(test_content),
        "image/png",
        original_filename: "minimal.png"
      )

      post file_hashes_upload_path, params: { file: upload }

      expect(response).to have_http_status(:created)
      json_response = JSON.parse(response.body)

      expected_hash = Digest::SHA256.hexdigest(test_content)
      expect(json_response["hash"]).to eq(expected_hash)
    end

    it "should handle duplicate file uploads" do
      test_content = File.binread(Rails.root.join("spec/fixtures/files/minimal.png"))

      upload1 = Rack::Test::UploadedFile.new(StringIO.new(test_content), "image/png", original_filename: "one.png")

      upload2 = Rack::Test::UploadedFile.new(StringIO.new(test_content), "image/png", original_filename: "two.png")

      # First upload
      post file_hashes_upload_path, params: { file: upload1 }
      expect(response).to have_http_status(:created)
      json_response1 = JSON.parse(response.body)
      hash1 = json_response1["hash"]

      # Second upload with same content
      post file_hashes_upload_path, params: { file: upload2 }
      expect(response).to have_http_status(:created)
      json_response2 = JSON.parse(response.body)
      hash2 = json_response2["hash"]

      # Hashes should be the same
      expect(hash1).to eq(hash2)

      # But only one record in database
      expect(FileHash.where(hash_value: hash1).count).to eq(1)
    end

    it "should save file to organized directory structure" do
      file = upload_minimal_png

      post file_hashes_upload_path, params: { file: file }

      expect(response).to have_http_status(:created)
      json_response = JSON.parse(response.body)
      saved_path = json_response["saved_at"]

      # Should be in storage/uploads/YYYY/MM/DD/ format
      expect(saved_path).to match(%r{storage/uploads/\d{4}/\d{2}/\d{2}/})

      # File should exist
      expect(File.exist?(Rails.root.join(saved_path))).to be true
    end

    it "should handle binary PNG files" do
      binary_content = File.binread(Rails.root.join("spec/fixtures/files/minimal.png"))
      file = Tempfile.new(["test", ".png"])
      file.binmode
      file.write(binary_content)
      file.rewind

      upload = Rack::Test::UploadedFile.new(file.path, "image/png")

      post file_hashes_upload_path, params: { file: upload }

      expect(response).to have_http_status(:created)
      json_response = JSON.parse(response.body)
      expect(json_response["hash"]).to be_present

      # Verify file was saved correctly
      saved_file_path = Rails.root.join(json_response["saved_at"])
      expect(File.exist?(saved_file_path)).to be true
      expect(File.size(saved_file_path)).to eq(binary_content.bytesize)

      file.close
      file.unlink
    end
  end

  describe "GET /file_hashes/:hash" do
    let(:known_sha256) { Digest::SHA256.hexdigest("known-hash-payload") }
    let(:unknown_sha256) { Digest::SHA256.hexdigest("unknown-entry") }

    before do
      FileHash.find_or_initialize_by(hash_value: known_sha256).tap do |fh|
        fh.ai_status = :ai_detected
        fh.save!
      end
    end

    it "returns status for an existing hash" do
      get "/file_hashes/#{known_sha256}"

      expect(response).to have_http_status(:ok)
      json = JSON.parse(response.body)
      expect(json["hash"]).to eq(known_sha256)
      expect(json["found_in_database"]).to be true
      expect(json["ai_status"]).to eq("ai_detected")
    end

    it "returns 404 with unknown ai_status when hash not in DB" do
      get "/file_hashes/#{unknown_sha256}"

      expect(response).to have_http_status(:not_found)
      json = JSON.parse(response.body)
      expect(json["hash"]).to eq(unknown_sha256)
      expect(json["found_in_database"]).to be false
      expect(json["ai_status"]).to eq("unknown")
    end

    it "returns 400 when hash is not a lowercase SHA-256 hex string" do
      get "/file_hashes/not-a-valid-sha256-hash-value-at-all"

      expect(response).to have_http_status(:bad_request)
      json = JSON.parse(response.body)
      expect(json["error"]).to eq("invalid hash")
    end
  end
end

