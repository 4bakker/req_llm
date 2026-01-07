defmodule ReqLLM.Streaming.Fixtures do
  @moduledoc false

  defmodule HTTPContext do
    @moduledoc """
    Lightweight HTTP context for streaming operations.

    This struct contains the minimal HTTP metadata needed for fixture capture
    and debugging, replacing the heavier Req.Request/Response structs for
    streaming operations.
    """

    @derive Jason.Encoder
    defstruct [
      :url,
      :method,
      :req_headers,
      :status,
      :resp_headers
    ]

    @type t :: %__MODULE__{
            url: String.t(),
            method: :get | :post | :put | :patch | :delete,
            req_headers: map(),
            status: integer() | nil,
            resp_headers: map() | nil
          }

    @doc """
    Creates a new HTTPContext from request parameters.
    """
    @spec new(String.t(), :get | :post | :put | :patch | :delete, map()) :: t()
    def new(url, method, headers) do
      %__MODULE__{
        url: url,
        method: method,
        req_headers: sanitize_headers(headers),
        status: nil,
        resp_headers: nil
      }
    end

    @doc """
    Updates the context with response status and headers.
    """
    @spec update_response(t(), integer(), map()) :: t()
    def update_response(%__MODULE__{} = context, status, headers) do
      %{context | status: status, resp_headers: sanitize_response_headers(headers)}
    end

    @doc """
    Builds HTTPContext from a Finch.Request struct.

    Extracts URL, method, and headers with proper sanitization.
    """
    @spec from_finch_request(Finch.Request.t()) :: t()
    def from_finch_request(%Finch.Request{} = finch_request) do
      url =
        if (finch_request.scheme == :https and finch_request.port == 443) or
             (finch_request.scheme == :http and finch_request.port == 80) do
          "#{finch_request.scheme}://#{finch_request.host}#{finch_request.path}"
        else
          "#{finch_request.scheme}://#{finch_request.host}:#{finch_request.port}#{finch_request.path}"
        end

      method = String.downcase(finch_request.method) |> String.to_atom()

      new(url, method, Map.new(finch_request.headers))
    end

    defp sanitize_headers(headers) when is_map(headers) do
      sensitive_keys = [
        "authorization",
        "x-api-key",
        "anthropic-api-key",
        "openai-api-key",
        "x-auth-token",
        "bearer",
        "api-key",
        "access-token"
      ]

      headers
      |> Map.new(fn {k, v} -> {String.downcase(to_string(k)), v} end)
      |> Enum.reduce(%{}, fn {key, value}, acc ->
        if key in sensitive_keys do
          Map.put(acc, key, "[REDACTED:#{key}]")
        else
          Map.put(acc, key, value)
        end
      end)
    end

    defp sanitize_headers(headers) when is_list(headers) do
      headers
      |> Map.new()
      |> sanitize_headers()
    end

    defp sanitize_headers(headers), do: headers

    defp sanitize_response_headers(headers) when is_map(headers) do
      headers
      |> Enum.map(fn {k, v} ->
        key = String.downcase(to_string(k))
        {k, sanitize_response_header_value(key, v)}
      end)
      |> Map.new()
    end

    defp sanitize_response_headers(headers) when is_list(headers) do
      headers
      |> Enum.map(fn
        {key, value} when is_binary(key) ->
          {key, sanitize_response_header_value(String.downcase(key), value)}

        {key, value} ->
          {to_string(key), sanitize_response_header_value(String.downcase(to_string(key)), value)}

        other ->
          other
      end)
    end

    defp sanitize_response_headers(headers), do: headers

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

    defp sanitize_response_header_value(_key, value), do: value
  end

  @doc """
  Extracts canonical JSON from Finch request body for fixture capture.

  Handles various body formats and returns a JSON-serializable structure.
  """
  @spec canonical_json_from_finch_request(Finch.Request.t()) :: map()
  def canonical_json_from_finch_request(%Finch.Request{body: body}) do
    case body do
      nil ->
        %{}

      binary when is_binary(binary) ->
        case Jason.decode(binary) do
          {:ok, json} -> json
          {:error, _} -> %{raw_body: binary}
        end

      {:stream, _} ->
        %{streaming_body: true}

      other ->
        %{unknown_body: inspect(other)}
    end
  rescue
    _ -> %{}
  end
end
