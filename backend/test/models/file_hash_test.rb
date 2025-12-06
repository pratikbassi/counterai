require "test_helper"

class FileHashTest < ActiveSupport::TestCase
  test "should require hash_value" do
    file_hash = FileHash.new
    assert_not file_hash.valid?
    assert_includes file_hash.errors[:hash_value], "can't be blank"
  end

  test "should enforce maximum length for hash_value" do
    file_hash = FileHash.new(hash_value: "a" * 256)
    assert_not file_hash.valid?
    assert_includes file_hash.errors[:hash_value], "is too long (maximum is 255 characters)"
  end

  test "should require unique hash_value" do
    FileHash.create!(hash_value: "unique_hash_123")
    
    duplicate = FileHash.new(hash_value: "unique_hash_123")
    assert_not duplicate.valid?
    assert_includes duplicate.errors[:hash_value], "has already been taken"
  end

  test "exist? should return true for existing hash" do
    hash_value = "existing_hash_456"
    FileHash.create!(hash_value: hash_value)
    
    result = FileHash.exist?([hash_value])
    assert_equal true, result[hash_value]
  end

  test "exist? should return false for non-existing hash" do
    hash_value = "non_existing_hash_789"
    
    result = FileHash.exist?([hash_value])
    assert_equal false, result[hash_value]
  end

  test "exist? should handle multiple hashes" do
    hash1 = "hash_one"
    hash2 = "hash_two"
    hash3 = "hash_three"
    
    FileHash.create!(hash_value: hash1)
    FileHash.create!(hash_value: hash2)
    # hash3 is not created
    
    result = FileHash.exist?([hash1, hash2, hash3])
    assert_equal true, result[hash1]
    assert_equal true, result[hash2]
    assert_equal false, result[hash3]
  end

  test "exist? should handle empty array" do
    result = FileHash.exist?([])
    assert_equal({}, result)
  end

  test "exist? should handle nil" do
    result = FileHash.exist?(nil)
    assert_equal({}, result)
  end

  test "exist? should handle single string" do
    hash_value = "single_hash"
    FileHash.create!(hash_value: hash_value)
    
    result = FileHash.exist?(hash_value)
    assert_equal true, result[hash_value]
  end

  test "exist? should remove duplicates" do
    hash_value = "duplicate_hash"
    FileHash.create!(hash_value: hash_value)
    
    result = FileHash.exist?([hash_value, hash_value, hash_value])
    assert_equal true, result[hash_value]
    assert_equal 1, result.keys.length
  end

  test "exist? should filter out nil and empty values" do
    hash_value = "valid_hash"
    FileHash.create!(hash_value: hash_value)
    
    result = FileHash.exist?([hash_value, nil, "", "   "])
    assert_equal true, result[hash_value]
    assert_equal 1, result.keys.length
  end

  test "exist? should be performant with many hashes" do
    # Create 100 hashes
    hashes = 100.times.map { |i| "hash_#{i}" }
    FileHash.insert_all(hashes.map { |h| { hash_value: h, created_at: Time.current, updated_at: Time.current } })
    
    # Check all 100 plus 50 non-existent
    check_hashes = hashes + 50.times.map { |i| "non_existing_#{i}" }
    
    result = FileHash.exist?(check_hashes)
    
    # All original hashes should exist
    hashes.each do |h|
      assert_equal true, result[h], "Hash #{h} should exist"
    end
    
    # All non-existing hashes should not exist
    50.times do |i|
      assert_equal false, result["non_existing_#{i}"], "Hash non_existing_#{i} should not exist"
    end
  end
end

