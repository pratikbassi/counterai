require "open3"

class DetectorJob < ApplicationJob
  queue_as :default

  CLASSIFY_SCRIPT = Rails.root.join("..", "model", "classify.py").to_s
  PYTHON_BIN = Rails.root.join("..", "model", ".venv", "bin", "python").to_s

  # @param file_address [String] Absolute path to the uploaded image
  # @param hash_value [String] SHA-256 hash identifying the FileHash record
  def perform(file_address, hash_value)
    record = FileHash.find_by(hash_value: hash_value)
    unless record
      Rails.logger.error "DetectorJob: No FileHash record for hash=#{hash_value}"
      return
    end

    unless File.exist?(file_address)
      Rails.logger.error "DetectorJob: File not found: #{file_address}"
      return
    end

    record.with_lock do
      record.reload
      unless record.unknown?
        Rails.logger.info "DetectorJob: skip duplicate/stale job hash=#{hash_value} status=#{record.ai_status}"
        return
      end

      Rails.logger.info "DetectorJob: Classifying #{file_address} (hash=#{hash_value})"

      stdout, stderr, status = Open3.capture3(
        PYTHON_BIN, CLASSIFY_SCRIPT, file_address, "--json", "--device", "cpu"
      )

      unless status.success?
        Rails.logger.error "DetectorJob: classify.py failed (exit #{status.exitstatus}): #{stderr}"
        return
      end

      result = begin
        JSON.parse(stdout)
      rescue JSON::ParserError => e
        Rails.logger.error "DetectorJob: invalid JSON from classify.py: #{e.message}"
        nil
      end
      return if result.nil?

      if result["error"]
        Rails.logger.error "DetectorJob: classify.py returned error: #{result['error']}"
        return
      end

      new_status = ai_status_from_label(result["label"])

      record.update!(ai_status: new_status)
      Rails.logger.info "DetectorJob: #{hash_value} → #{new_status} (#{result['label']} p=#{result['confidence']})"
    end
  end

  private

  # Map model label to FileHash ai_status.
  # Handles both named labels ("Real"/"Fake") and numeric ("0"/"1")
  # where 0 = real, 1 = fake (training convention).
  AI_DETECTED_LABELS = %w[1 fake ai synthetic generated].freeze

  def ai_status_from_label(label)
    AI_DETECTED_LABELS.any? { |tok| label.to_s.downcase.include?(tok) } ? :ai_detected : :ai_not_detected
  end
end
