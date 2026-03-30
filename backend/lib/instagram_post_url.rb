# frozen_string_literal: true

require "json"
require "uri"

# Shared rules with config/instagram_post_url.json (also read by the Python downloader).
module InstagramPostUrl
  DEFAULT_RULES = {
    "hosts" => %w[instagram.com instagr.am],
    "post_path_regex" => "^/(?:p|reel|tv)/[A-Za-z0-9_-]+/?$"
  }.freeze

  class << self
    def config_path
      raw = ENV["INSTAGRAM_POST_URL_CONFIG"].to_s.strip
      if raw.present?
        File.expand_path(raw)
      else
        Rails.root.join("..", "config", "instagram_post_url.json").to_s
      end
    end

    def match?(url)
      uri = URI.parse(url.to_s.strip)
      return false unless uri.is_a?(URI::HTTP) && %w[http https].include?(uri.scheme)
      return false if uri.host.blank?

      host = uri.host.to_s.downcase.delete_prefix("www.")
      return false unless hosts.include?(host)

      path_regex.match?(uri.path.to_s)
    rescue URI::InvalidURIError
      false
    end

    def reload!
      @rules = nil
      @hosts = nil
      @path_regex = nil
      self
    end

    private

    def rules
      @rules ||= load_rules
    end

    def load_rules
      path = config_path
      return DEFAULT_RULES unless File.file?(path)

      JSON.parse(File.read(path))
    rescue JSON::ParserError, Errno::ENOENT, Errno::EACCES
      DEFAULT_RULES
    end

    def hosts
      @hosts ||= Array(rules["hosts"]).map { |h| h.to_s.downcase.delete_prefix("www.") }.freeze
    end

    def path_regex
      @path_regex ||= Regexp.new(rules.fetch("post_path_regex", DEFAULT_RULES["post_path_regex"]), Regexp::IGNORECASE)
    end
  end
end
