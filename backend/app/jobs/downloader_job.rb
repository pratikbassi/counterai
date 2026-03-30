require "digest"
require "fileutils"
require "open3"
require "pathname"

class DownloaderJob < ApplicationJob
  queue_as :default

  DOWNLOAD_SCRIPT = Rails.root.join("..", "downloader", "download_url.py").to_s
  DEFAULT_PYTHON = Rails.root.join("..", "downloader", ".venv", "bin", "python").to_s
  DEFAULT_TIMEOUT_SEC = 120
  DEFAULT_MAX_URL_BYTES = 8192

  # @param url [String] HTTP(S) Instagram post, reel, or TV permalink only (see config/instagram_post_url.json)
  def perform(url)
    url = url.to_s.strip
    if url.blank?
      Rails.logger.error "DownloaderJob: empty URL"
      return
    end

    max_bytes = max_url_bytes
    if url.bytesize > max_bytes
      Rails.logger.error "DownloaderJob: URL too long (#{url.bytesize} bytes, max #{max_bytes})"
      return
    end

    unless InstagramPostUrl.match?(url)
      Rails.logger.info "DownloaderJob: skipped (not an Instagram post/reel/tv URL): #{url.truncate(200)}"
      return
    end

    Rails.logger.info "DownloaderJob: downloading #{url}"

    stdout, stderr, status = capture_downloader(url)

    if status.nil?
      Rails.logger.error "DownloaderJob: #{stderr.presence || 'subprocess failed or timed out'}"
      return
    end

    parsed = parse_downloader_stdout(stdout)

    unless status.success?
      detail = parsed&.dig("error").presence || stderr.presence || stdout.presence || "no output"
      Rails.logger.error "DownloaderJob: download_url.py exit #{status.exitstatus}: #{detail}"
      return
    end

    unless parsed
      Rails.logger.error(
        "DownloaderJob: invalid JSON on success (stderr: #{stderr.presence || 'empty'}): " \
        "#{stdout.truncate(500)}"
      )
      return
    end

    unless truthy?(parsed["ok"])
      Rails.logger.error "DownloaderJob: #{parsed['error'].presence || 'download failed'}"
      return
    end

    finalize_download!(parsed)
  end

  private

  def truthy?(v)
    v == true || v.to_s == "true"
  end

  def python_executable
    ENV["DOWNLOADER_PYTHON"].presence || DEFAULT_PYTHON
  end

  def downloader_output_dir
    raw = ENV["DOWNLOADER_OUTPUT_DIR"].to_s.strip
    if raw.present?
      File.expand_path(raw)
    else
      Rails.root.join("storage", "uploads").to_s
    end
  end

  def max_url_bytes
    v = ENV["DOWNLOADER_MAX_URL_BYTES"]
    return DEFAULT_MAX_URL_BYTES if v.blank?
    Integer(v)
  rescue ArgumentError
    DEFAULT_MAX_URL_BYTES
  end

  def downloader_timeout_sec
    Integer(ENV.fetch("DOWNLOADER_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC.to_s))
  rescue ArgumentError
    DEFAULT_TIMEOUT_SEC
  end

  def parse_downloader_stdout(raw)
    return nil if raw.blank?
    try_parse = ->(s) { JSON.parse(s) }
    try_parse.call(raw.strip)
  rescue JSON::ParserError
    line = raw.lines.map(&:strip).reverse.find(&:presence)
    return nil if line.blank?
    JSON.parse(line)
  rescue JSON::ParserError
    nil
  end

  def finalize_download!(parsed)
    saved_path = parsed["saved_path"].to_s.strip
    if saved_path.blank?
      Rails.logger.error "DownloaderJob: missing saved_path in downloader output"
      return
    end

    path = Pathname.new(saved_path)
    path = path.expand_path unless path.absolute?
    unless path.file?
      Rails.logger.error "DownloaderJob: saved file missing: #{path}"
      return
    end

    if path.symlink?
      Rails.logger.error "DownloaderJob: refusing symlink path: #{path}"
      return
    end

    unless path_under_uploads?(path)
      Rails.logger.error "DownloaderJob: refusing path outside storage/uploads: #{path}"
      return
    end

    expected = parsed["sha256"].to_s.strip.downcase
    disk_hash = Digest::SHA256.file(path).hexdigest
    if expected.present? && disk_hash != expected
      Rails.logger.error "DownloaderJob: SHA-256 mismatch (disk #{disk_hash} vs script #{expected})"
      return
    end

    found_before = FileHash.exists?(hash_value: disk_hash)
    record = FileHash.find_or_create_by!(hash_value: disk_hash)
    absolute = path.to_s

    if record.unknown?
      DetectorJob.perform_later(absolute, disk_hash)
      Rails.logger.info(
        "DownloaderJob: stored #{disk_hash[0, 16]}… (#{parsed['bytes']} bytes, #{parsed['source']}) " \
        "found_in_database=#{found_before}; enqueued DetectorJob"
      )
    else
      Rails.logger.info(
        "DownloaderJob: stored #{disk_hash[0, 16]}… (#{parsed['bytes']} bytes, #{parsed['source']}) " \
        "found_in_database=#{found_before}; ai_status=#{record.ai_status} (detection skipped)"
      )
    end
  rescue ActiveRecord::RecordInvalid => e
    Rails.logger.error "DownloaderJob: FileHash save failed: #{e.message}"
  end

  def uploads_realpath
    @uploads_realpath ||= begin
      dir = Rails.root.join("storage", "uploads")
      FileUtils.mkdir_p(dir)
      dir.realpath
    end
  end

  def path_under_uploads?(path)
    base = uploads_realpath
    return false unless base
    real = path.realpath
    real.to_s.start_with?(base.to_s + File::SEPARATOR) || real == base
  rescue Errno::ENOENT
    false
  end

  # Returns [stdout, stderr, Process::Status] or [stdout, stderr, nil] on timeout / spawn failure.
  def capture_downloader(url)
    child_env = { "DOWNLOADER_OUTPUT_DIR" => downloader_output_dir.to_s }
    cmd = [python_executable, DOWNLOAD_SCRIPT, url, "--json"]
    Open3.popen3(child_env, *cmd) do |stdin, stdout_io, stderr_io, wait_thr|
      stdin.close
      out_thread = Thread.new { stdout_io.read }
      err_thread = Thread.new { stderr_io.read }
      if wait_thr.join(downloader_timeout_sec)
        status = wait_thr.value
        [out_thread.value, err_thread.value, status]
      else
        kill_downloader_process(wait_thr)
        stdout_str = out_thread.value
        stderr_str = err_thread.value
        combined = [stderr_str, "timed out after #{downloader_timeout_sec}s"].compact_blank.join("\n")
        [stdout_str, combined.presence || "timed out", nil]
      end
    end
  rescue Errno::ENOENT => e
    ["", "cannot start downloader (#{e.message})", nil]
  end

  def kill_downloader_process(wait_thr)
    pid = wait_thr.pid
    Process.kill("TERM", pid)
    return if wait_thr.join(5)
    Process.kill("KILL", pid)
  rescue Errno::ESRCH
    nil
  end
end
