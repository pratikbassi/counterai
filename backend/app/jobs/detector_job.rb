class DetectorJob < ApplicationJob
  queue_as :default

  # Perform the detection job
  # @param file_address [String] The file path or address to process
  # @return [String] Returns "completed" after processing
  def perform(file_address)
    Rails.logger.info "DetectorJob: Starting detection for file: #{file_address}"
    
    # Sleep for 3 seconds as specified
    sleep(3)
    
    Rails.logger.info "DetectorJob: Completed detection for file: #{file_address}"
    
    # Return "completed" result
    "completed"
  end
end



