require "open3"
require "timeout"

class DetectorJob < ApplicationJob
  queue_as :default

  # Promoted checkpoint (Phase G6 winner — efficientnet_v2_s @ 288, seed 42,
  # val macro_f1 0.9808 on the merged ~140k holdout; see
  # `model/docs/MODEL_ABLATION_PLAN.md` Phase G / G6 for provenance).
  # Production should pin an explicit stamp so routine training runs that
  # overwrite `artifacts/best_real_fake.pt` do not silently change the
  # deployed model.
  DEFAULT_CHECKPOINT_FILENAME = "best_real_fake_20260422_002356_seed42.pt".freeze

  MODEL_ROOT = Rails.root.join("..", "model").freeze

  # Frozen at class load so we don't re-resolve env vars on every job.
  PYTHON_BIN = (ENV["CLASSIFIER_PYTHON"].presence ||
                MODEL_ROOT.join(".venv", "bin", "python").to_s).freeze
  CLASSIFY_SCRIPT = (ENV["CLASSIFIER_SCRIPT"].presence ||
                     MODEL_ROOT.join("classify.py").to_s).freeze
  CHECKPOINT_PATH = (ENV["CLASSIFIER_CHECKPOINT"].presence ||
                     MODEL_ROOT.join("artifacts", DEFAULT_CHECKPOINT_FILENAME).to_s).freeze
  DEVICE = (ENV["CLASSIFIER_DEVICE"].presence || "cpu").freeze

  # Read each call so tests and boot-time ENV changes stay predictable.
  def self.classifier_timeout_sec
    Integer(ENV.fetch("CLASSIFIER_TIMEOUT_SEC", "60"))
  end

  # @param file_address [String] Absolute path to the uploaded image
  # @param hash_value [String] SHA-256 hash identifying the FileHash record
  def perform(file_address, hash_value)
    record = FileHash.find_by(hash_value: hash_value)
    unless record
      Rails.logger.error "DetectorJob: No FileHash record for hash=#{hash_value}"
      return
    end

    unless File.exist?(file_address)
      Rails.logger.error "DetectorJob: File not found basename=#{File.basename(file_address.to_s)} hash=#{hash_value}"
      return
    end

    record.with_lock do
      record.reload
      unless record.unknown?
        Rails.logger.info "DetectorJob: skip duplicate/stale job hash=#{hash_value} status=#{record.ai_status}"
        return
      end

      ckpt_basename = File.basename(CHECKPOINT_PATH)
      Rails.logger.info "DetectorJob: Classifying basename=#{File.basename(file_address)} hash=#{hash_value} ckpt=#{ckpt_basename}"

      stdout, stderr, status = nil
      begin
        Timeout.timeout(self.class.classifier_timeout_sec) do
          stdout, stderr, status = Open3.capture3(
            PYTHON_BIN, CLASSIFY_SCRIPT, file_address,
            "--json",
            "--checkpoint", CHECKPOINT_PATH,
            "--device", DEVICE
          )
        end
      rescue Timeout::Error
        Rails.logger.error "DetectorJob: classify.py timeout after #{self.class.classifier_timeout_sec}s hash=#{hash_value} ckpt=#{ckpt_basename}"
        return
      end

      unless status.success?
        preview = stderr.to_s.byteslice(0, 200)
        Rails.logger.error "DetectorJob: classify.py failed (exit #{status.exitstatus}) hash=#{hash_value} ckpt=#{ckpt_basename} stderr_preview=#{preview.inspect}"
        return
      end

      result = begin
        JSON.parse(stdout)
      rescue JSON::ParserError => e
        Rails.logger.error "DetectorJob: invalid JSON from classify.py: #{e.message} hash=#{hash_value}"
        nil
      end
      return if result.nil?

      if result["error"]
        Rails.logger.error "DetectorJob: classify.py returned error hash=#{hash_value} error=#{result['error'].to_s.byteslice(0, 200).inspect}"
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
