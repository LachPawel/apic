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
use futures::StreamExt;

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

    fn make_request(&self, text: &str) -> CreateChatCompletionRequest {
        CreateChatCompletionRequest {
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
        }
    }

    /// Non-streaming: awaits the full response. Used by benchmarks.
    pub async fn send_message(&self, text: &str) -> Result<String> {
        let response = self.inner.chat().create(self.make_request(text)).await?;
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();
        Ok(content)
    }

    /// Streaming: sends complete sentences to `tx` as they arrive.
    ///
    /// Splits on `. `, `! `, `? ` boundaries so TTS can start on the
    /// first sentence while the LLM is still generating the rest.
    pub async fn stream_sentences(
        &self,
        text: &str,
        tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Result<()> {
        let mut request = self.make_request(text);
        request.stream = Some(true);

        let mut stream = self.inner.chat().create_stream(request).await?;
        let mut buf = String::new();

        while let Some(result) = stream.next().await {
            let response = result.map_err(|e| anyhow::anyhow!("LLM stream error: {e}"))?;
            for choice in &response.choices {
                if let Some(ref token) = choice.delta.content {
                    buf.push_str(token);
                    // Flush complete sentences immediately.
                    while let Some(pos) = sentence_boundary(&buf) {
                        let sentence = buf[..pos].trim().to_string();
                        buf = buf[pos..].trim_start().to_string();
                        if !sentence.is_empty() {
                            let _ = tx.send(sentence);
                        }
                    }
                }
            }
        }

        // Flush any remaining text as a final sentence.
        let tail = buf.trim().to_string();
        if !tail.is_empty() {
            let _ = tx.send(tail);
        }

        Ok(())
    }
}

/// Returns the byte offset just past the first sentence-ending punctuation
/// followed by whitespace (or end of string).
fn sentence_boundary(s: &str) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        if matches!(ch, '.' | '!' | '?') {
            let next = chars.get(i + 1);
            if next.is_none_or(|&c| c == ' ' || c == '\n') {
                // byte offset = sum of char lengths up to and including i
                let byte_pos = s
                    .char_indices()
                    .nth(i + 1)
                    .map(|(b, _)| b)
                    .unwrap_or(s.len());
                return Some(byte_pos);
            }
        }
    }
    None
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

    #[test]
    fn sentence_boundary_splits_on_period_space() {
        // "Hello world." — period at byte 11, boundary is byte 12 (exclusive)
        assert_eq!(sentence_boundary("Hello world. How are you?"), Some(12));
    }

    #[test]
    fn sentence_boundary_returns_none_for_no_boundary() {
        assert_eq!(sentence_boundary("Hello world"), None);
    }

    #[test]
    fn sentence_boundary_handles_end_of_string() {
        assert_eq!(sentence_boundary("Done."), Some(5));
    }
}
