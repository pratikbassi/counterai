require "rails_helper"

RSpec.describe FileHash, type: :model do
  describe "validations" do
    it "should require hash_value" do
      file_hash = FileHash.new
      expect(file_hash).not_to be_valid
      expect(file_hash.errors[:hash_value]).to include("can't be blank")
    end

    it "should enforce maximum length for hash_value" do
      file_hash = FileHash.new(hash_value: "a" * 256)
      expect(file_hash).not_to be_valid
      expect(file_hash.errors[:hash_value]).to include("is too long (maximum is 255 characters)")
    end

    it "should require unique hash_value" do
      FileHash.create!(hash_value: "unique_hash_123")
      
      duplicate = FileHash.new(hash_value: "unique_hash_123")
      expect(duplicate).not_to be_valid
      expect(duplicate.errors[:hash_value]).to include("has already been taken")
    end
  end

  describe ".exist?" do
    it "should return true for existing hash" do
      hash_value = "existing_hash_456"
      FileHash.create!(hash_value: hash_value)
      
      result = FileHash.exist?([hash_value])
      expect(result[hash_value]).to eq(true)
    end

    it "should return false for non-existing hash" do
      hash_value = "non_existing_hash_789"
      
      result = FileHash.exist?([hash_value])
      expect(result[hash_value]).to eq(false)
    end

    it "should handle multiple hashes" do
      hash1 = "hash_one"
      hash2 = "hash_two"
      hash3 = "hash_three"
      
      FileHash.create!(hash_value: hash1)
      FileHash.create!(hash_value: hash2)
      # hash3 is not created
      
      result = FileHash.exist?([hash1, hash2, hash3])
      expect(result[hash1]).to eq(true)
      expect(result[hash2]).to eq(true)
      expect(result[hash3]).to eq(false)
    end

    it "should handle empty array" do
      result = FileHash.exist?([])
      expect(result).to eq({})
    end

    it "should handle nil" do
      result = FileHash.exist?(nil)
      expect(result).to eq({})
    end

    it "should handle single string" do
      hash_value = "single_hash"
      FileHash.create!(hash_value: hash_value)
      
      result = FileHash.exist?(hash_value)
      expect(result[hash_value]).to eq(true)
    end

    it "should remove duplicates" do
      hash_value = "duplicate_hash"
      FileHash.create!(hash_value: hash_value)
      
      result = FileHash.exist?([hash_value, hash_value, hash_value])
      expect(result[hash_value]).to eq(true)
      expect(result.keys.length).to eq(1)
    end

    it "should filter out nil and empty values" do
      hash_value = "valid_hash"
      FileHash.create!(hash_value: hash_value)
      
      result = FileHash.exist?([hash_value, nil, "", "   "])
      expect(result[hash_value]).to eq(true)
      expect(result.keys.length).to eq(1)
    end

    it "should be performant with many hashes" do
      # Create 100 hashes
      hashes = 100.times.map { |i| "hash_#{i}" }
      FileHash.insert_all(hashes.map { |h| { hash_value: h, created_at: Time.current, updated_at: Time.current } })
      
      # Check all 100 plus 50 non-existent
      check_hashes = hashes + 50.times.map { |i| "non_existing_#{i}" }
      
      result = FileHash.exist?(check_hashes)
      
      # All original hashes should exist
      hashes.each do |h|
        expect(result[h]).to eq(true), "Hash #{h} should exist"
      end
      
      # All non-existing hashes should not exist
      50.times do |i|
        expect(result["non_existing_#{i}"]).to eq(false), "Hash non_existing_#{i} should not exist"
      end
    end
  end
end

