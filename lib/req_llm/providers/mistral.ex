defmodule ReqLLM.Providers.Mistral do
  @moduledoc """
  Mistral AI provider – OpenAI-compatible chat API.

  ## Implementation

  Uses built-in OpenAI-style encoding/decoding defaults with custom handling for:
  - Tool calls that may omit the "type" field (defaults to "function")
  - Arguments that may be returned as a map OR a JSON string

  ## Mistral-Specific Parameters

  Mistral supports additional parameters via `:provider_options`:
  - `:random_seed` - Deterministic sampling (Mistral uses this instead of OpenAI's `seed`)
  - `:safe_prompt` - Inject a safety prompt before all conversations
  - `:prediction` - Enable speculative decoding with expected content

  ## Configuration

      # Add to .env file (automatically loaded)
      MISTRAL_API_KEY=...
  """

  use ReqLLM.Provider,
    id: :mistral,
    default_base_url: "https://api.mistral.ai/v1",
    default_env_key: "MISTRAL_API_KEY"

  use ReqLLM.Provider.Defaults

  @impl ReqLLM.Provider
  def decode_response({req, resp}) do
    case resp.status do
      200 ->
        # Pre-process the response to normalize Mistral's tool_calls format
        body = ensure_parsed_body(resp.body)
        normalized_body = normalize_mistral_tool_calls(body)
        ReqLLM.Provider.Defaults.default_decode_response({req, %{resp | body: normalized_body}})

      _ ->
        ReqLLM.Provider.Defaults.default_decode_response({req, resp})
    end
  end

  # Mistral's tool_calls may differ from OpenAI in two ways:
  # 1. They don't include "type": "function" - we add it
  # 2. Arguments may be a map (already parsed) instead of a JSON string - we stringify it
  defp normalize_mistral_tool_calls(body) when is_map(body) do
    case body do
      %{"choices" => choices} when is_list(choices) ->
        normalized_choices =
          Enum.map(choices, fn choice ->
            case choice do
              %{"message" => %{"tool_calls" => tool_calls} = message} when is_list(tool_calls) ->
                normalized_tool_calls = Enum.map(tool_calls, &normalize_tool_call/1)
                %{choice | "message" => %{message | "tool_calls" => normalized_tool_calls}}

              _ ->
                choice
            end
          end)

        %{body | "choices" => normalized_choices}

      _ ->
        body
    end
  end

  defp normalize_mistral_tool_calls(body), do: body

  defp normalize_tool_call(%{"id" => _, "function" => function} = tc) do
    # Add type: "function" if not present
    tc = Map.put_new(tc, "type", "function")

    # Normalize arguments: if it's a map, encode to JSON string for consistency
    case function do
      %{"arguments" => args} when is_map(args) ->
        normalized_function = %{function | "arguments" => Jason.encode!(args)}
        %{tc | "function" => normalized_function}

      _ ->
        tc
    end
  end

  defp normalize_tool_call(other), do: other

  defp ensure_parsed_body(body) when is_binary(body) do
    case Jason.decode(body) do
      {:ok, parsed} -> parsed
      {:error, _} -> body
    end
  end

  defp ensure_parsed_body(body), do: body
end
