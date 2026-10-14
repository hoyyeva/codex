use super::normalize_ollama_history;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

fn items(value: Value) -> Vec<ResponseItem> {
    serde_json::from_value(value).unwrap()
}

fn checkpoint(summary: &str, retained: Value) -> Value {
    json!({
        "type": "compaction",
        "encrypted_content": json!({
            "type": "ollama_compaction", "version": 1,
            "summary": summary, "retained": retained
        }).to_string()
    })
}

#[test]
fn removes_ollama_reasoning_without_changing_other_history() {
    let preserved = items(json!([
        {"type": "reasoning", "id": "rs_67ab0123456789", "summary": [], "encrypted_content": "openai-state"},
        {"type": "reasoning", "id": "rs_12", "summary": [{"type": "summary_text", "text": "readable"}], "encrypted_content": "opaque-state"},
        {"type": "compaction", "encrypted_content": "openai-checkpoint"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Remember amber-42"}]},
        {"type": "custom_tool_call", "call_id": "call_1", "name": "read", "namespace": "workspace.tools", "input": "notes.txt"},
        {"type": "custom_tool_call_output", "call_id": "call_1", "output": "tulip-73"}
    ]));
    for id in ["rs_43372", "rs_resp_123456"] {
        let mut input = preserved.clone();
        input.insert(2, serde_json::from_value(json!({
            "type": "reasoning", "id": id, "summary": [{"type": "summary_text", "text": "Ollama thinking"}], "encrypted_content": "Ollama thinking"
        })).unwrap());
        assert_eq!(normalize_ollama_history(input).unwrap(), preserved);
    }
}

#[test]
fn expands_summary_and_retained_tool_history_in_place() {
    let prefix = json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Original question"}]});
    let suffix = json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Continue"}]});
    let input = items(json!([
        prefix,
        checkpoint(
            "Remember kiwi-42",
            json!([
                {"role": "assistant", "content": "Checking", "thinking": "private continuation", "tool_calls": [{"id": "call_note", "function": {"index": 0, "name": "workspace.tools.read_note", "arguments": {"input": "notes.txt"}}}]},
                {"role": "tool", "content": "kiwi-42", "tool_call_id": "call_note"}
            ])
        ),
        suffix
    ]));
    assert_eq!(
        normalize_ollama_history(input).unwrap(),
        items(json!([
            prefix,
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Remember kiwi-42"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking"}]},
            {"type": "function_call", "call_id": "call_note", "name": "workspace.tools.read_note", "arguments": "{\"input\":\"notes.txt\"}"},
            {"type": "function_call_output", "call_id": "call_note", "output": "kiwi-42"},
            suffix
        ]))
    );
}

#[test]
fn preserves_named_tool_results_and_images() {
    let image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZkAAAAASUVORK5CYII=";
    let input = items(json!([{
        "type": "compaction", "encrypted_content": json!({
            "type": "ollama_compaction", "version": 1, "summary": "Screenshot context",
            "retained": [{"role": "tool", "content": "image result", "images": [image], "tool_name": "workspace.tools.read"}],
            "standalone_names": {"0": {"name": "read", "namespace": "workspace.tools"}}
        }).to_string()
    }]));
    assert_eq!(
        normalize_ollama_history(input).unwrap(),
        items(json!([
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Screenshot context"}]},
            {"type": "function_call_output", "name": "read", "namespace": "workspace.tools", "output": [
                {"type": "input_text", "text": "image result"},
                {"type": "input_image", "image_url": format!("data:image/png;base64,{image}"), "detail": "auto"}
            ]}
        ]))
    );
}

#[test]
fn uses_newest_ollama_checkpoint_and_preserves_openai_state() {
    let foreign = json!({"type": "compaction", "encrypted_content": "openai-state"});
    let message = json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Keep this"}]});
    assert_eq!(
        normalize_ollama_history(items(json!([
            checkpoint("old", Value::Null),
            foreign,
            message,
            checkpoint("current", json!([]))
        ])))
        .unwrap(),
        items(json!([
            foreign, message,
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "current"}]}
        ]))
    );
}

#[test]
fn rejects_unsupported_checkpoints_instead_of_losing_context() {
    for payload in [
        json!({"type": "ollama_compaction", "version": 2, "summary": "future"}),
        json!({"type": "ollama_compaction", "version": 1, "summary": ""}),
        json!({"type": "ollama_compaction", "version": 1, "summary": "context", "new_field": "unhandled"}),
        json!({"type": "ollama_compaction", "version": 1, "summary": "context", "retained": [{"role": "tool", "content": "missing identity"}]}),
        json!({"type": "ollama_compaction", "version": 1, "summary": "context", "retained": [{"role": "user", "content": "image", "images": ["invalid"]}]}),
    ] {
        let input =
            items(json!([{"type": "compaction", "encrypted_content": payload.to_string()}]));
        assert!(
            normalize_ollama_history(input).is_err(),
            "accepted {payload}"
        );
    }
}
