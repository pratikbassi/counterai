# frozen_string_literal: true

# Shared configuration at repository root (sibling of `backend/`).
repo_env = Rails.root.parent.join(".env")
if repo_env.file?
  require "dotenv"
  Dotenv.load(repo_env.to_s)
end
