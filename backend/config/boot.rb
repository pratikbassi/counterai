ENV["BUNDLE_GEMFILE"] ||= File.expand_path("../Gemfile", __dir__)

require "bundler/setup" # Set up gems listed in the Gemfile.
require "bootsnap/setup" # Speed up boot time by caching expensive operations.

# Load `.env` before Active Record reads `database.yml` (PGPASSWORD, etc.).
# `overwrite: true` so file values replace stale/empty vars already in the shell (default Dotenv.load does not).
begin
  require "dotenv"
  backend_root = File.expand_path("..", __dir__)
  repo_root = File.expand_path("..", backend_root)
  Dotenv.load(
    File.join(repo_root, ".env"),
    File.join(backend_root, ".env"),
    overwrite: true
  )
rescue LoadError
end
