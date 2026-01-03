defmodule ReqLLM.Providers.MistralTest do
  @moduledoc """
  Provider-level tests for Mistral implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Mistral

  alias ReqLLM.Context
  alias ReqLLM.Providers.Mistral

  # Helper to create a Mistral model struct directly (LLMDB may not have Mistral models yet)
  defp mistral_model(model_id \\ "mistral-large-latest") do
    %LLMDB.Model{
      id: model_id,
      provider: :mistral
    }
  end

  describe "provider contract" do
    test "provider identity and configuration" do
      assert Mistral.provider_id() == :mistral
      assert Mistral.base_url() == "https://api.mistral.ai/v1"
      assert Mistral.default_env_key() == "MISTRAL_API_KEY"
    end

    test "provider schema separation from core options" do
      schema_keys = Mistral.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "provider schema combined with generation schema includes all core keys" do
      full_schema = Mistral.provider_extended_generation_schema()
      full_keys = Keyword.keys(full_schema.schema)
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- full_keys
      assert missing == [], "Missing core generation keys in extended schema: #{inspect(missing)}"
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      model = mistral_model()
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Mistral.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/chat/completions"
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      model = mistral_model()
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> Mistral.attach(model, opts)

      # Verify authentication
      auth_header = Enum.find(request.headers, fn {name, _} -> name == "authorization" end)
      assert auth_header != nil
      {_, [auth_value]} = auth_header
      assert String.starts_with?(auth_value, "Bearer ")

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      model = mistral_model()
      prompt = "Hello world"

      # Unsupported operation
      {:error, error} = Mistral.prepare_request(:unsupported, model, prompt, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      wrong_model = %LLMDB.Model{id: "gpt-4", provider: :openai}

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> Mistral.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      model = mistral_model()
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.id,
          stream: false
        ]
      }

      updated_request = Mistral.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["model"] == "mistral-large-latest"
      assert is_list(decoded["messages"])
      assert length(decoded["messages"]) == 2
      assert decoded["stream"] == false
      refute Map.has_key?(decoded, "tools")

      [system_msg, user_msg] = decoded["messages"]
      assert system_msg["role"] == "system"
      assert user_msg["role"] == "user"
    end

    test "encode_body with tools" do
      model = mistral_model()
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            name: [type: :string, required: true, doc: "A name parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.id,
          stream: false,
          tools: [tool]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1

      [encoded_tool] = decoded["tools"]
      assert encoded_tool["function"]["name"] == "test_tool"
    end

    test "encode_body handles standard OpenAI options" do
      model = mistral_model()
      context = context_fixture()

      test_cases = [
        {[temperature: 0.2, max_tokens: 55, top_p: 0.9],
         fn json ->
           assert json["temperature"] == 0.2
           assert json["max_tokens"] == 55
           assert json["top_p"] == 0.9
         end},
        {[presence_penalty: 0.2, user: "test_user", seed: 12_345],
         fn json ->
           assert json["presence_penalty"] == 0.2
           assert json["user"] == "test_user"
           assert json["seed"] == 12_345
         end}
      ]

      for {options, assertion} <- test_cases do
        full_options = [context: context, model: model.id, stream: false] ++ options
        mock_request = %Req.Request{options: full_options}
        updated_request = Mistral.encode_body(mock_request)
        decoded = Jason.decode!(updated_request.body)
        assertion.(decoded)
      end
    end
  end

  describe "response decoding & normalization" do
    test "decode_response handles non-streaming responses" do
      model = mistral_model()
      mock_json_response = openai_format_json_fixture(model: "mistral-large-latest")

      mock_resp = %Req.Response{
        status: 200,
        body: mock_json_response
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, id: "mistral:mistral-large-latest"]
      }

      {req, resp} = Mistral.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert is_binary(response.id)
      assert response.model == model.id
      assert response.stream? == false

      # Verify message normalization
      assert response.message.role == :assistant
      text = ReqLLM.Response.text(response)
      assert is_binary(text)
      assert String.length(text) > 0
      assert response.finish_reason in [:stop, :length]

      # Verify usage normalization
      assert is_integer(response.usage.input_tokens)
      assert is_integer(response.usage.output_tokens)
      assert is_integer(response.usage.total_tokens)

      # Verify context advancement (original + assistant)
      assert length(response.context.messages) == 3
      assert List.last(response.context.messages).role == :assistant
    end

    test "decode_response handles API errors with non-200 status" do
      error_body = %{
        "error" => %{
          "message" => "Invalid API key",
          "type" => "authentication_error",
          "code" => "invalid_api_key"
        }
      }

      mock_resp = %Req.Response{
        status: 401,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, id: "mistral-large-latest"]
      }

      {req, error} = Mistral.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 401
      assert error.reason =~ " API error"
      assert error.response_body == error_body
    end
  end

  describe "usage extraction" do
    test "extract_usage with valid usage data" do
      model = mistral_model()

      body_with_usage = %{
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 20,
          "total_tokens" => 30
        }
      }

      {:ok, usage} = Mistral.extract_usage(body_with_usage, model)
      assert usage["prompt_tokens"] == 10
      assert usage["completion_tokens"] == 20
      assert usage["total_tokens"] == 30
    end

    test "extract_usage with missing usage data" do
      model = mistral_model()
      body_without_usage = %{"choices" => []}

      {:error, :no_usage_found} = Mistral.extract_usage(body_without_usage, model)
    end

    test "extract_usage with invalid body type" do
      model = mistral_model()

      {:error, :invalid_body} = Mistral.extract_usage("invalid", model)
      {:error, :invalid_body} = Mistral.extract_usage(nil, model)
      {:error, :invalid_body} = Mistral.extract_usage(123, model)
    end
  end

  describe "error handling & robustness" do
    test "context validation" do
      # Multiple system messages should fail
      invalid_context =
        Context.new([
          Context.system("System 1"),
          Context.system("System 2"),
          Context.user("Hello")
        ])

      assert_raise ReqLLM.Error.Validation.Error,
                   ~r/Context should have at most one system message/,
                   fn ->
                     Context.validate!(invalid_context)
                   end
    end
  end
end
