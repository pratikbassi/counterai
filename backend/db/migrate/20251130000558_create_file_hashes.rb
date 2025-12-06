class CreateFileHashes < ActiveRecord::Migration[8.1]
  def change
    create_table :file_hashes do |t|
      t.string :hash_value, null: false, limit: 255
      t.timestamps null: false
    end

    add_index :file_hashes, :hash_value, unique: true
  end
end
