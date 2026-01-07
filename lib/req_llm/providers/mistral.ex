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

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  @provider_schema [
    random_seed: [
      type: :integer,
      doc: "Deterministic sampling for reproducible outputs"
    ],
    safe_prompt: [
      type: :boolean,
      doc: "Inject Mistral's safety prompt before all conversations"
    ],
    prediction: [
      type: :map,
      doc: "Enable speculative decoding with expected content"
    ]
  ]

  @impl ReqLLM.Provider
  def encode_body(request) do
    request = ReqLLM.Provider.Defaults.default_encode_body(request)
    body = Jason.decode!(request.body)

    provider_opts = request.options[:provider_options] || []

    body =
      body
      |> translate_tool_choice_format()
      |> maybe_put(:random_seed, provider_opts[:random_seed])
      |> maybe_put(:safe_prompt, provider_opts[:safe_prompt])
      |> maybe_put(:prediction, provider_opts[:prediction])

    encoded_body = Jason.encode!(body)
    Map.put(request, :body, encoded_body)
  end

  defp translate_tool_choice_format(body) do
    {tool_choice, body_key} =
      cond do
        Map.has_key?(body, :tool_choice) -> {Map.get(body, :tool_choice), :tool_choice}
        Map.has_key?(body, "tool_choice") -> {Map.get(body, "tool_choice"), "tool_choice"}
        true -> {nil, nil}
      end

    {type, name} =
      if is_map(tool_choice) do
        {Map.get(tool_choice, :type) || Map.get(tool_choice, "type"),
         Map.get(tool_choice, :name) || Map.get(tool_choice, "name")}
      else
        {nil, nil}
      end

    if type == "tool" && name do
      replacement =
        if is_map_key(tool_choice, :type) do
          %{type: "function", function: %{name: name}}
        else
          %{"type" => "function", "function" => %{"name" => name}}
        end

      Map.put(body, body_key, replacement)
    else
      body
    end
  end

  @impl ReqLLM.Provider
  def translate_options(_operation, _model, opts) do
    {seed, opts} = Keyword.pop(opts, :seed)

    opts =
      if seed do
        provider_opts = Keyword.get(opts, :provider_options, [])
        provider_opts = Keyword.put(provider_opts, :random_seed, seed)
        Keyword.put(opts, :provider_options, provider_opts)
      else
        opts
      end

    {opts, []}
  end

  @doc """
  Custom attach_stream that ensures translate_options is called for streaming requests.

  This is necessary because the default streaming path doesn't call translate_options,
  which means seed -> random_seed translation wouldn't be applied to streaming requests.
  """
  @impl ReqLLM.Provider
  def attach_stream(model, context, opts, finch_name) do
    {translated_opts, _warnings} = translate_options(:chat, model, opts)
    base_url = ReqLLM.Provider.Options.effective_base_url(__MODULE__, model, translated_opts)
    opts_with_base_url = Keyword.put(translated_opts, :base_url, base_url)

    ReqLLM.Provider.Defaults.default_attach_stream(
      __MODULE__,
      model,
      context,
      opts_with_base_url,
      finch_name
    )
  end

  @impl ReqLLM.Provider
  def decode_stream_event(%{data: data} = event, model) when is_map(data) do
    normalized_event = %{event | data: normalize_streaming_tool_calls(data)}
    ReqLLM.Provider.Defaults.default_decode_stream_event(normalized_event, model)
  end

  def decode_stream_event(event, model) do
    ReqLLM.Provider.Defaults.default_decode_stream_event(event, model)
  end

  defp normalize_streaming_tool_calls(%{"choices" => choices} = data) when is_list(choices) do
    normalized_choices =
      Enum.map(choices, fn
        %{"delta" => %{"tool_calls" => tool_calls} = delta} = choice when is_list(tool_calls) ->
          normalized_tool_calls = Enum.map(tool_calls, &normalize_streaming_tool_call/1)
          %{choice | "delta" => %{delta | "tool_calls" => normalized_tool_calls}}

        choice ->
          choice
      end)

    %{data | "choices" => normalized_choices}
  end

  defp normalize_streaming_tool_calls(data), do: data

  defp normalize_streaming_tool_call(%{"function" => _} = tc) do
    Map.put_new(tc, "type", "function")
  end

  defp normalize_streaming_tool_call(other), do: other

  @impl ReqLLM.Provider
  def decode_response({req, resp}) do
    case resp.status do
      200 ->
        body = ensure_parsed_body(resp.body)

        normalized_body =
          body
          |> normalize_mistral_tool_calls()
          |> normalize_mistral_finish_reason()

        ReqLLM.Provider.Defaults.default_decode_response({req, %{resp | body: normalized_body}})

      _ ->
        ReqLLM.Provider.Defaults.default_decode_response({req, resp})
    end
  end

  defp normalize_mistral_finish_reason(%{"choices" => choices} = body) when is_list(choices) do
    normalized_choices =
      Enum.map(choices, fn
        %{"finish_reason" => "model_length"} = choice ->
          %{choice | "finish_reason" => "length"}

        choice ->
          choice
      end)

    %{body | "choices" => normalized_choices}
  end

  defp normalize_mistral_finish_reason(body), do: body

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
