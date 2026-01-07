defmodule Mix.Tasks.ReqLlm.FixtureSanitize do
  @shortdoc "Sanitize sensitive data from existing fixture files"
  @moduledoc """
  Re-sanitize existing fixture files to remove sensitive data.

  This task reads all fixture files and sanitizes response headers that may
  contain sensitive information like:
  - Cookies (set-cookie headers)
  - Organization/project identifiers
  - Rate limit information
  - Correlation/request IDs

  ## Usage

      mix req_llm.fixture_sanitize                    # Sanitize all fixtures
      mix req_llm.fixture_sanitize --dry-run          # Show what would be changed
      mix req_llm.fixture_sanitize openai             # Sanitize fixtures for specific provider
      mix req_llm.fixture_sanitize openai gpt_4       # Sanitize fixtures for specific model

  ## Flags

      --dry-run        Show what would be changed without modifying files
      --verbose        Show detailed information about each file processed
  """

  use Mix.Task

  @fixture_root Path.expand("test/support/fixtures")

  @impl Mix.Task
  def run(args) do
    {opts, args, _} =
      OptionParser.parse(args,
        switches: [dry_run: :boolean, verbose: :boolean],
        aliases: [v: :verbose, d: :dry_run]
      )

    dry_run? = Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)

    if dry_run? do
      Mix.shell().info("DRY RUN MODE - No files will be modified")
    end

    paths = build_fixture_paths(args)

    if Enum.empty?(paths) do
      Mix.shell().error("No fixture files found")
      System.halt(1)
    end

    Mix.shell().info("Found #{length(paths)} fixture file(s) to process")

    results =
      Enum.map(paths, fn path ->
        sanitize_fixture_file(path, dry_run?, verbose?)
      end)

    modified = Enum.count(results, & &1)
    unchanged = length(results) - modified

    Mix.shell().info("")
    Mix.shell().info("Summary:")
    Mix.shell().info("  Modified: #{modified}")
    Mix.shell().info("  Unchanged: #{unchanged}")

    if dry_run? and modified > 0 do
      Mix.shell().info("")
      Mix.shell().info("Run without --dry-run to apply changes")
    end
  end

  defp build_fixture_paths([]) do
    Path.wildcard(Path.join(@fixture_root, "**/*.json"))
  end

  defp build_fixture_paths([provider]) do
    Path.wildcard(Path.join(@fixture_root, "#{provider}/**/*.json"))
  end

  defp build_fixture_paths([provider, model]) do
    Path.wildcard(Path.join(@fixture_root, "#{provider}/#{model}/*.json"))
  end

  defp build_fixture_paths(_) do
    Mix.shell().error("Invalid arguments. Expected: [provider] [model]")
    System.halt(1)
  end

  defp sanitize_fixture_file(path, dry_run?, verbose?) do
    content = File.read!(path)
    json = Jason.decode!(content)

    sanitized = sanitize_fixture(json)

    if sanitized == json do
      if verbose? do
        Mix.shell().info("  Unchanged: #{Path.relative_to_cwd(path)}")
      end

      false
    else
      if verbose? do
        Mix.shell().info("  Modified: #{Path.relative_to_cwd(path)}")
      end

      if !dry_run? do
        sanitized_json = Jason.encode!(sanitized, pretty: true)
        File.write!(path, sanitized_json)
      end

      true
    end
  rescue
    e ->
      Mix.shell().error("Error processing #{Path.relative_to_cwd(path)}: #{inspect(e)}")
      false
  end

  defp sanitize_fixture(%{"response" => response} = fixture) when is_map(response) do
    sanitized_response =
      response
      |> sanitize_response_headers()
      |> sanitize_response_body()

    %{fixture | "response" => sanitized_response}
  end

  defp sanitize_fixture(fixture), do: fixture

  defp sanitize_response_headers(%{"headers" => headers} = response) when is_map(headers) do
    sanitized_headers =
      Enum.reduce(headers, %{}, fn {key, value}, acc ->
        key_str = String.downcase(to_string(key))
        sanitized_value = sanitize_response_header_value(key_str, value)
        Map.put(acc, key, sanitized_value)
      end)

    %{response | "headers" => sanitized_headers}
  end

  defp sanitize_response_headers(%{"headers" => headers} = response) when is_list(headers) do
    sanitized_headers =
      Enum.map(headers, fn
        {key, value} when is_binary(key) ->
          {key, sanitize_response_header_value(String.downcase(key), value)}

        {key, value} ->
          {to_string(key), sanitize_response_header_value(String.downcase(to_string(key)), value)}

        other ->
          other
      end)

    %{response | "headers" => sanitized_headers}
  end

  defp sanitize_response_headers(response), do: response

  defp sanitize_response_header_value(key, value) when is_binary(key) do
    cond do
      key == "set-cookie" ->
        if is_list(value) do
          Enum.map(value, fn _ -> "[REDACTED:set-cookie]" end)
        else
          "[REDACTED:set-cookie]"
        end

      key in ["openai-organization", "openai-project"] ->
        "[REDACTED:#{key}]"

      String.starts_with?(key, "x-ratelimit-remaining-") ->
        if is_list(value) do
          Enum.map(value, fn _ -> "[REDACTED]" end)
        else
          "[REDACTED]"
        end

      key in [
        "mistral-correlation-id",
        "x-kong-request-id",
        "x-request-id",
        "cf-ray"
      ] ->
        if is_list(value) do
          Enum.map(value, fn _ -> "[REDACTED:#{key}]" end)
        else
          "[REDACTED:#{key}]"
        end

      true ->
        value
    end
  end

  defp sanitize_response_body(%{"body" => body} = response) when is_map(body) do
    sanitized_body = sanitize_json(body)
    %{response | "body" => sanitized_body}
  end

  defp sanitize_response_body(response), do: response

  defp sanitize_json(%{} = map) do
    Enum.reduce(map, %{}, fn {k, v}, acc ->
      key = String.downcase(to_string(k))

      sanitized_value =
        if key in ["id", "request_id", "correlation_id"] do
          "[REDACTED:#{k}]"
        else
          sanitize_json(v)
        end

      Map.put(acc, k, sanitized_value)
    end)
  end

  defp sanitize_json(list) when is_list(list) do
    Enum.map(list, &sanitize_json/1)
  end

  defp sanitize_json(other), do: other
end
