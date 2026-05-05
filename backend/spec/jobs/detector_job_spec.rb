require "rails_helper"

RSpec.describe DetectorJob do
  let(:good_status) do
    instance_double(Process::Status, success?: true, exitstatus: 0)
  end
  let(:bad_status) do
    instance_double(Process::Status, success?: false, exitstatus: 1)
  end
  let(:fixture_image_path) { Rails.root.join("spec/fixtures/files/minimal.png").to_s }

  def perform_with_env(timeout_sec: nil)
    prev = ENV["CLASSIFIER_TIMEOUT_SEC"]
    ENV["CLASSIFIER_TIMEOUT_SEC"] = timeout_sec.to_s if timeout_sec
    yield
  ensure
    if prev.nil?
      ENV.delete("CLASSIFIER_TIMEOUT_SEC")
    else
      ENV["CLASSIFIER_TIMEOUT_SEC"] = prev
    end
  end

  describe "#perform" do
    let(:hash_value) { Digest::SHA256.hexdigest("spec-image-bytes") }
    let!(:record) { FileHash.create!(hash_value: hash_value, ai_status: :unknown) }

    context "successful classification" do
      it "sets ai_not_detected for Real label" do
        allow(Open3).to receive(:capture3).and_return(
          ['{"label":"Real","confidence":0.9}', "", good_status]
        )

        described_class.perform_now(fixture_image_path, hash_value)

        expect(record.reload.ai_status).to eq("ai_not_detected")
      end

      it "sets ai_detected for Fake label" do
        allow(Open3).to receive(:capture3).and_return(
          ['{"label":"Fake","confidence":0.8}', "", good_status]
        )

        described_class.perform_now(fixture_image_path, hash_value)

        expect(record.reload.ai_status).to eq("ai_detected")
      end
    end

    context "classifier exits non-zero" do
      it "keeps ai_status unknown" do
        allow(Open3).to receive(:capture3).and_return(["", "boom", bad_status])

        described_class.perform_now(fixture_image_path, hash_value)

        expect(record.reload.ai_status).to eq("unknown")
      end
    end

    context "invalid JSON stdout" do
      it "keeps ai_status unknown" do
        allow(Open3).to receive(:capture3).and_return(["not json", "", good_status])

        described_class.perform_now(fixture_image_path, hash_value)

        expect(record.reload.ai_status).to eq("unknown")
      end
    end

    context "JSON error payload" do
      it "keeps ai_status unknown when error key present" do
        allow(Open3).to receive(:capture3).and_return(
          ['{"error":"bad image"}', "", good_status]
        )

        described_class.perform_now(fixture_image_path, hash_value)

        expect(record.reload.ai_status).to eq("unknown")
      end
    end

    context "missing FileHash record" do
      it "returns without invoking classifier" do
        allow(Open3).to receive(:capture3)

        ghost = "f" * 64
        expect do
          described_class.perform_now(fixture_image_path, ghost)
        end.not_to raise_error

        expect(Open3).not_to have_received(:capture3)
      end
    end

    context "missing file" do
      it "keeps ai_status unknown and does not call classifier" do
        allow(Open3).to receive(:capture3)

        described_class.perform_now("/nonexistent/#{SecureRandom.hex(8)}.png", hash_value)

        expect(Open3).not_to have_received(:capture3)
        expect(record.reload.ai_status).to eq("unknown")
      end
    end

    context "timeout" do
      it "keeps ai_status unknown when classify exceeds CLASSIFIER_TIMEOUT_SEC" do
        perform_with_env(timeout_sec: 1) do
          allow(Open3).to receive(:capture3) { sleep 3 }

          described_class.perform_now(fixture_image_path, hash_value)

          expect(record.reload.ai_status).to eq("unknown")
        end
      end
    end

    context "non-unknown record skipped" do
      before { record.update!(ai_status: :ai_detected) }

      it "does not invoke classifier" do
        allow(Open3).to receive(:capture3)

        described_class.perform_now(fixture_image_path, hash_value)

        expect(Open3).not_to have_received(:capture3)
      end
    end
  end
end
