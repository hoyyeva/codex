use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use serde::Deserialize;
use serde::de::Error as _;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// Makes Ollama continuation state readable by OpenAI without changing saved history.
pub fn normalize_ollama_history(input: Vec<ResponseItem>) -> serde_json::Result<Vec<ResponseItem>> {
    let mut output = Vec::with_capacity(input.len());
    let mut compaction = None;
    for item in input {
        match &item {
            ResponseItem::Reasoning {
                id: Some(id),
                summary,
                encrypted_content: Some(encrypted),
                ..
            } if id.as_str().strip_prefix("rs_").is_some_and(|suffix| {
                let suffix = suffix.strip_prefix("resp_").unwrap_or(suffix);
                (1..=6).contains(&suffix.len()) && suffix.bytes().all(|byte| byte.is_ascii_digit())
            }) && !encrypted.is_empty()
                && summary
                    .iter()
                    .map(|part| {
                        let ReasoningItemReasoningSummary::SummaryText { text } = part;
                        text.as_str()
                    })
                    .collect::<String>()
                    == *encrypted => {}
            ResponseItem::Compaction {
                encrypted_content, ..
            } if encrypted_content.trim_start().starts_with('{') => {
                let payload: Value = serde_json::from_str(encrypted_content)?;
                if payload.get("type").and_then(Value::as_str) == Some("ollama_compaction") {
                    let payload: OllamaCompaction = serde_json::from_value(payload)?;
                    // A later Ollama checkpoint supersedes the earlier Ollama checkpoint.
                    compaction = Some((output.len(), payload.into_items()?));
                } else {
                    output.push(item);
                }
            }
            _ => output.push(item),
        }
    }
    if let Some((index, items)) = compaction {
        output.splice(index..index, items);
    }
    Ok(output)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OllamaCompaction {
    #[serde(rename = "type")]
    kind: String,
    version: u32,
    summary: String,
    #[serde(default)]
    retained: Option<Vec<OllamaMessage>>,
    #[serde(default)]
    standalone_names: HashMap<usize, FunctionName>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FunctionName {
    name: String,
    #[serde(default)]
    namespace: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(default, rename = "thinking")]
    _thinking: String,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
    #[serde(default)]
    tool_name: String,
    #[serde(default)]
    tool_call_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OllamaToolCall {
    id: String,
    function: OllamaFunction,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OllamaFunction {
    #[serde(default, rename = "index")]
    _index: usize,
    name: String,
    arguments: serde_json::Map<String, Value>,
}

impl OllamaCompaction {
    fn into_items(self) -> serde_json::Result<Vec<ResponseItem>> {
        if self.kind != "ollama_compaction" || self.version != 1 || self.summary.trim().is_empty() {
            return Err(serde_json::Error::custom(
                "unsupported or empty Ollama compaction checkpoint",
            ));
        }
        let retained = self.retained.unwrap_or_default();
        for (&index, name) in &self.standalone_names {
            let qualified = if name.namespace.is_empty() {
                name.name.clone()
            } else {
                format!("{}.{}", name.namespace, name.name)
            };
            if name.name.trim().is_empty()
                || !retained.get(index).is_some_and(|message| {
                    message.role == "tool"
                        && message.tool_call_id.is_empty()
                        && message.tool_name == qualified
                })
            {
                return Err(serde_json::Error::custom(
                    "invalid retained Ollama tool identity",
                ));
            }
        }
        let mut items = vec![serde_json::from_value(json!({
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": self.summary}]
        }))?];
        for (index, message) in retained.into_iter().enumerate() {
            let mut content = vec![json!({"type": "input_text", "text": message.content})];
            for image in &message.images {
                let bytes =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, image)
                        .map_err(serde_json::Error::custom)?;
                let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                    "image/png"
                } else if bytes.starts_with(b"\xff\xd8\xff") {
                    "image/jpeg"
                } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
                    "image/webp"
                } else {
                    return Err(serde_json::Error::custom(
                        "unsupported retained Ollama image",
                    ));
                };
                content.push(json!({"type": "input_image", "image_url": format!("data:{mime};base64,{image}"), "detail": "auto"}));
            }
            if message.role == "tool" {
                let output = if message.images.is_empty() {
                    json!(message.content)
                } else {
                    json!(content)
                };
                let item = if message.tool_call_id.is_empty() {
                    let name = self.standalone_names.get(&index).ok_or_else(|| {
                        serde_json::Error::custom("retained Ollama tool output has no identity")
                    })?;
                    let mut item = json!({"type": "function_call_output", "name": name.name, "output": output});
                    if !name.namespace.is_empty() {
                        item["namespace"] = json!(name.namespace);
                    }
                    item
                } else if message.tool_name == "tool_search" {
                    if !message.images.is_empty() {
                        return Err(serde_json::Error::custom(
                            "retained Ollama tool search contains images",
                        ));
                    }
                    let tools: Vec<Value> = serde_json::from_str(&message.content)?;
                    json!({"type": "tool_search_output", "call_id": message.tool_call_id, "execution": "client", "status": "completed", "tools": tools})
                } else {
                    json!({"type": "function_call_output", "call_id": message.tool_call_id, "output": output})
                };
                items.push(serde_json::from_value(item)?);
            } else {
                if !matches!(
                    message.role.as_str(),
                    "user" | "assistant" | "system" | "developer"
                ) {
                    return Err(serde_json::Error::custom(
                        "invalid retained Ollama message role",
                    ));
                }
                if !message.content.is_empty()
                    || !message.images.is_empty()
                    || message.tool_calls.is_empty()
                {
                    if message.role == "assistant" {
                        content[0]["type"] = json!("output_text");
                    }
                    items.push(serde_json::from_value(
                        json!({"type": "message", "role": message.role, "content": content}),
                    )?);
                }
            }
            for call in message.tool_calls {
                if call.id.is_empty() || call.function.name.is_empty() {
                    return Err(serde_json::Error::custom(
                        "retained Ollama tool call has no identity",
                    ));
                }
                let item = if call.function.name == "tool_search" {
                    json!({"type": "tool_search_call", "call_id": call.id, "execution": "client", "status": "completed", "arguments": call.function.arguments})
                } else {
                    json!({"type": "function_call", "call_id": call.id, "name": call.function.name, "arguments": serde_json::to_string(&call.function.arguments)?})
                };
                items.push(serde_json::from_value(item)?);
            }
        }
        Ok(items)
    }
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod tests;
