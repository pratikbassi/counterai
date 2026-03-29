require "open3"

class DownloaderJob < ApplicationJob
  queue_as :default

  DOWNLOAD_SCRIPT = Rails.root.join("..", "downloader", "download_url.py").to_s
  PYTHON_BIN = Rails.root.join("..", "downloader", ".venv", "bin", "python").to_s

  # @param url [String] HTTP(S) URL pasted by the user (image or page — script decides)
  def perform(url)
    url = url.to_s.strip
    if url.blank?
      Rails.logger.error "DownloaderJob: empty URL"
      return
    end

    Rails.logger.info "DownloaderJob: placeholder download for #{url}"

    stdout, stderr, status = Open3.capture3(
      PYTHON_BIN, DOWNLOAD_SCRIPT, url, "--json"
    )

    result = JSON.parse(stdout)
    unless status.success?
      Rails.logger.error(
        "DownloaderJob: download_url.py exit #{status.exitstatus}: " \
        "#{result['error'].presence || stderr.presence || stdout}"
      )
      return
    end

    Rails.logger.info "DownloaderJob: #{result['status']} — #{result['message']}"
  rescue JSON::ParserError
    Rails.logger.error "DownloaderJob: invalid JSON from script: #{stdout}"
  end
end
