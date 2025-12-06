class FileHash < ApplicationRecord
  validates :hash_value, presence: true, uniqueness: true, length: { maximum: 255 }

  # Class method to check which hashes exist in the database
  # Returns a hash mapping hash => boolean
  def self.exist?(hashes)
    return {} if hashes.blank?

    # Normalize input to array
    hash_array = hashes.is_a?(Array) ? hashes : [hashes]
    
    # Remove duplicates and nil/empty values
    hash_array = hash_array.compact.reject(&:blank?).uniq

    return {} if hash_array.empty?

    # Single query to find all existing hashes
    existing_hashes = where(hash_value: hash_array).pluck(:hash_value).to_set

    # Return hash mapping each input hash to whether it exists
    hash_array.index_with { |h| existing_hashes.include?(h) }
  end
end

