use std::fs;
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Serialize)]
struct OllamaOptions {
    num_ctx: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_ollama_parsing() {
        // 1. Start a local mock server
        let server = MockServer::start();

        // 2. Prepare a fake Ollama response
        let fake_response = json!({
            "model": "gemini-3-flash-preview:cloud",
            "created_at": "2026-01-09T20:40:11Z",
            "response": "### Fix List\n* Use std::print instead of printf",
            "done": true
        });

        // 3. Setup the mock behavior
        let ollama_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/api/generate")
                .header("Content-Type", "application/json");
            then.status(200)
                .json_body(fake_response);
        });

        // 4. Point our request logic to the mock server instead of localhost:11434
        // (Note: You may need to update fetch_review to accept a URL string)
        let mock_url = format!("{}/api/generate", server.base_url());

        // Simulating the logic of fetch_review
        let client = reqwest::Client::new();
        let res = client.post(mock_url)
            .json(&json!({ "model": "gemma3", "prompt": "test", "stream": false }))
            .send()
            .await
            .unwrap();

        let data: OllamaResponse = res.json().await.unwrap();

        // 5. Assertions
        ollama_mock.assert(); // Ensure the mock was actually called
        assert!(data.response.contains("std::print"));
        println!("Test passed: Parsed mock Ollama response correctly.");
    }
}

async fn fetch_review(code: String) -> Result<String, Box<dyn std::error::Error>> {
    // Increased timeout to 10 minutes for safety
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let prompt = format!("Review this C++23 code. Update @instructions.md tasks:\n{}", code);

   let body = OllamaRequest {
        model: "gemma3".to_string(),
        prompt,
        stream: false,
        options: OllamaOptions {
            // Set this slightly higher than your 'prompt=' log value
            num_ctx: 49152,
            temperature: 0.1,
        },
    };

    println!("-> Sending request to Ollama (context)...");
    let res = client.post("http://localhost:11434/api/generate")
        .json(&body)
        .send()
        .await?;

    if !res.status().is_success() {
    println!("Ollama returned error:");
        return Err(format!("Ollama returned error: {}", res.status()).into());
    }

    println!("Ollama returned ok:");

    let data = res.json::<OllamaResponse>().await?;
    Ok(data.response)
}

fn get_all_cpp(root: &str) -> String {
    let mut code = String::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        if p.extension().map_or(false, |ext| ext == "cpp" || ext == "hpp") {
            if let Ok(c) = fs::read_to_string(p) {
                code.push_str(&format!("\n// File: {:?}\n{}", p, c));
            }
        }
    }
    code
}

#[tokio::main]
async fn main() {
    let repo = "../";
    let target = "../instructions.md";

    loop {
        println!("\n Checking for changes...");
        let code = get_all_cpp(repo);

        if code.is_empty() {
            println!("! No C++ files found in {}", repo);
        } else {
            match fetch_review(code).await {
                Ok(review) => {
                    fs::write(target, review).unwrap();
                    println!("✔ @instructions.md updated.");
                }
                Err(e) => eprintln!("✘ Error: {}", e),
            }
        }

        println!("Waiting 60s...");
        sleep(Duration::from_secs(60)).await;
    }
}