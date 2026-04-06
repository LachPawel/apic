use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    },
    Client,
};

const APFEL_BASE_URL: &str = "http://127.0.0.1:11434/v1";
const APFEL_MODEL: &str = "apple-foundationmodel";
const SYSTEM_PROMPT: &str = "You are a concise voice assistant. Reply in one or two sentences.";

pub struct LlmClient {
    inner: Client<OpenAIConfig>,
}

impl LlmClient {
    pub fn new() -> Self {
        Self::with_base_url(APFEL_BASE_URL)
    }

    pub fn with_base_url(base_url: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("unused");
        Self {
            inner: Client::with_config(config),
        }
    }

    pub async fn send_message(&self, text: &str) -> Result<String> {
        let request = CreateChatCompletionRequest {
            model: APFEL_MODEL.to_string(),
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(
                        SYSTEM_PROMPT.to_string(),
                    ),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(text.to_string()),
                    name: None,
                }),
            ],
            ..Default::default()
        };

        let response = self.inner.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }
}

impl Default for LlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn send_message_returns_assistant_content() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "apple-foundationmodel",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28
                }
            })))
            .mount(&mock_server)
            .await;

        let client = LlmClient::with_base_url(&mock_server.uri());
        let result = client
            .send_message("What is the capital of France?")
            .await;

        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert_eq!(result.unwrap(), "The capital of France is Paris.");
    }
}
