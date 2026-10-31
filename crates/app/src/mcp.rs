//! A Model Context Protocol server: each [`agent`](crate::agent) op is a tool, so an MCP
//! client (Claude Code, Claude Desktop, any agent framework) can paint with wetpaint.
//! Messages are JSON-RPC 2.0, one per line, on stdin and stdout (the MCP stdio transport).
//! `render` replies carry the PNG as an image the model can look at.

use std::io::{self, BufRead, Write};

use serde_json::{Map, Value, json};

use crate::agent::OPS;

/// Protocol versions this server speaks, newest first; a client asking for another gets
/// the newest.
const VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// A rendered image is scaled down to this many pixels on its longer side unless the call
/// asks for another size, so it fits in the model's context.
const RENDER_MAX: u64 = 1024;

const INSTRUCTIONS: &str = "wetpaint is a wet-media painting app. Coordinates are canvas pixels from the top-left. \
Typical session: `state` (size, layers), `set` brush.tool / brush.color / brush.size, `stroke` with points, `step` to let \
the paint flow and dry (25 steps a second), `render` to look. `describe` lists every command and setting; `command` runs \
any menu or keyboard command by name.";

/// The tool list: one tool per op.
pub fn tools() -> Value {
    let tools: Vec<Value> = OPS
        .iter()
        .map(|op| {
            let props: Map<String, Value> = op
                .args
                .iter()
                .map(|(name, ty, doc)| {
                    let mut p = Map::new();
                    if *ty != "any" {
                        p.insert("type".into(), (*ty).into());
                    }
                    if !doc.is_empty() {
                        p.insert("description".into(), (*doc).into());
                    }
                    (name.to_string(), Value::Object(p))
                })
                .collect();
            json!({ "name": op.name, "description": op.doc, "inputSchema": { "type": "object", "properties": props } })
        })
        .collect();
    json!({ "tools": tools })
}

/// The reply to one JSON-RPC message (`None` for notifications), calling `op` for tools;
/// `quit` is set when the session should end.
pub fn reply(msg: &Value, op: &mut dyn FnMut(&Value) -> Value, quit: &mut bool) -> Option<Value> {
    let id = msg.get("id").cloned()?;
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(VERSIONS[0]);
            let version = VERSIONS.iter().find(|v| **v == asked).unwrap_or(&VERSIONS[0]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "wetpaint", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tools()),
        "tools/call" => call_tool(&params, op, quit),
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    })
}

fn call_tool(params: &Value, op: &mut dyn FnMut(&Value) -> Value, quit: &mut bool) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).ok_or((-32602, "tools/call needs a name".to_owned()))?;
    if !OPS.iter().any(|o| o.name == name) {
        return Err((-32602, format!("unknown tool: {name}")));
    }
    let mut req = match params.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    req.insert("op".into(), name.into());
    if name == "render" && !req.contains_key("path") && !req.contains_key("max_size") {
        req.insert("max_size".into(), RENDER_MAX.into());
    }
    let mut r = op(&Value::Object(req));
    let ok = r["ok"] == true;
    *quit |= ok && name == "quit";
    let mut content = Vec::new();
    if let Some(png) = r.as_object_mut().and_then(|m| m.remove("png")) {
        content.push(json!({ "type": "image", "data": png, "mimeType": "image/png" }));
    }
    content.insert(0, json!({ "type": "text", "text": r.to_string() }));
    Ok(json!({ "content": content, "isError": !ok }))
}

/// Serve MCP on `input` / `output` until the input ends or `quit` is called.
pub fn serve(op: &mut dyn FnMut(&Value) -> Value, input: impl BufRead, mut output: impl Write) -> io::Result<()> {
    let mut quit = false;
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => reply(&msg, op, &mut quit),
            Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": format!("parse error: {e}") } })),
        };
        if let Some(r) = reply {
            writeln!(output, "{r}")?;
            output.flush()?;
        }
        if quit {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_client_lists_tools_paints_and_sees_the_painting() {
        let mut app = crate::agent::headless(96, 64);
        let mut op = |r: &Value| crate::agent::handle(&mut app, r);
        let input = [
            json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "test", "version": "0" } } }),
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
            json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "stroke", "arguments": { "points": [[10, 30], [80, 30]] } } }),
            json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": "render", "arguments": {} } }),
            json!({ "jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": { "name": "set", "arguments": { "key": "brush.nope", "value": 1 } } }),
            json!({ "jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": { "name": "quit", "arguments": { "discard": true } } }),
            json!({ "jsonrpc": "2.0", "id": 7, "method": "ping" }),
        ]
        .map(|m| m.to_string())
        .join("\n");
        let mut out = Vec::new();
        serve(&mut op, input.as_bytes(), &mut out).unwrap();
        let replies: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(replies.len(), 6, "no reply to the notification, none after quit");
        assert_eq!(replies[0]["result"]["protocolVersion"], "2025-06-18");
        let tools = replies[1]["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t["name"] == "stroke" && t["inputSchema"]["properties"]["points"]["type"] == "array"));
        assert_eq!(replies[2]["result"]["isError"], false);
        let image = &replies[3]["result"]["content"][1];
        assert_eq!(image["type"], "image");
        assert!(image["data"].as_str().unwrap().len() > 100);
        assert_eq!(replies[4]["result"]["isError"], true);
        assert_eq!(replies[5]["id"], 6);
    }
}
