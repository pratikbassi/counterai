# frozen_string_literal: true

class AddAiStatusToFileHashes < ActiveRecord::Migration[8.1]
  def change
    add_column :file_hashes, :ai_status, :string, null: false, default: "unknown"
    add_index :file_hashes, :ai_status
  end
end
