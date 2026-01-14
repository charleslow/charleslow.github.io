use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::process;

#[derive(Debug, Deserialize, Serialize)]
struct Book {
    #[serde(alias = "items")]
    sections: Vec<BookItem>,
    #[serde(flatten)]
    other: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum BookItem {
    Chapter { Chapter: Chapter },
    Separator(Value),
}

#[derive(Debug, Deserialize, Serialize)]
struct Chapter {
    content: String,
    sub_items: Vec<BookItem>,
    #[serde(flatten)]
    other: serde_json::Map<String, Value>,
}

/// Transform text inside <<text>> to add color
/// Pattern: (?<!<)<<(?!<)(.+?)(?<!>)>>(?!>)
/// This matches << >> but not <<< >>> (avoids triple angle brackets)
fn transform_text(input_text: &str) -> String {
    // Rust regex doesn't support lookbehind, so we use a different approach
    // We'll match <<...>> and check context manually
    let pattern = Regex::new(r"<<(.+?)>>").unwrap();
    
    let mut result = String::new();
    let mut last_end = 0;
    
    for caps in pattern.captures_iter(input_text) {
        let m = caps.get(0).unwrap();
        let start = m.start();
        let end = m.end();
        
        // Check for triple angle brackets (lookbehind/lookahead simulation)
        let preceded_by_lt = start > 0 && input_text.as_bytes().get(start - 1) == Some(&b'<');
        let followed_by_gt = end < input_text.len() && input_text.as_bytes().get(end) == Some(&b'>');
        
        if preceded_by_lt || followed_by_gt {
            // This is part of <<< or >>>, don't transform
            result.push_str(&input_text[last_end..end]);
        } else {
            // Transform this match
            result.push_str(&input_text[last_end..start]);
            let inner = &caps[1];
            result.push_str(&format!(r#"<span style="color:orange">{}</span>"#, inner));
        }
        last_end = end;
    }
    
    result.push_str(&input_text[last_end..]);
    result
}

/// Find all arXiv URLs in the text
fn find_arxiv_urls(text: &str) -> Vec<String> {
    let pattern = Regex::new(r"https?://arxiv\.org/abs/\d+\.\d+").unwrap();
    pattern
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

fn process_chapter(chapter: &mut Chapter, arxiv_urls: &mut Vec<String>) {
    chapter.content = transform_text(&chapter.content);
    arxiv_urls.extend(find_arxiv_urls(&chapter.content));
    
    for sub_item in &mut chapter.sub_items {
        if let BookItem::Chapter { Chapter: sub_chapter } = sub_item {
            sub_chapter.content = transform_text(&sub_chapter.content);
            arxiv_urls.extend(find_arxiv_urls(&sub_chapter.content));
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    // Handle "supports" command
    if args.len() > 1 && args[1] == "supports" {
        process::exit(0);
    }
    
    // Read JSON from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    
    // Parse as [context, book] tuple
    let parsed: Value = serde_json::from_str(&input)
        .expect("Failed to parse JSON from stdin");
    
    let array = parsed.as_array()
        .expect("Expected JSON array [context, book]");
    
    if array.len() < 2 {
        eprintln!("Expected [context, book] array");
        process::exit(1);
    }
    
    // Parse the book portion
    let mut book: Book = serde_json::from_value(array[1].clone())
        .expect("Failed to parse book structure");
    
    let mut arxiv_urls: Vec<String> = Vec::new();
    
    // Process all chapters
    for item in &mut book.sections {
        if let BookItem::Chapter { Chapter: chapter } = item {
            process_chapter(chapter, &mut arxiv_urls);
        }
    }
    
    // Write arXiv URLs to file (sorted and deduplicated)
    let unique_urls: BTreeSet<String> = arxiv_urls.into_iter().collect();
    let mut file = File::create("arxiv_urls.txt")?;
    for url in unique_urls {
        writeln!(file, "{}", url)?;
    }
    
    // Output modified book to stdout - preserve the wrapper structure if present
    // Output modified book to stdout
    let output = serde_json::to_string(&book)
        .expect("Failed to serialize book");
    print!("{}", output);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_text() {
        assert_eq!(
            transform_text("Hello <<world>>!"),
            r#"Hello <span style="color:orange">world</span>!"#
        );
        
        // Should not transform triple brackets
        assert_eq!(
            transform_text("Hello <<<world>>>!"),
            "Hello <<<world>>>!"
        );
        
        // Multiple transforms
        assert_eq!(
            transform_text("<<a>> and <<b>>"),
            r#"<span style="color:orange">a</span> and <span style="color:orange">b</span>"#
        );
    }

    #[test]
    fn test_find_arxiv_urls() {
        let text = "Check out https://arxiv.org/abs/2301.12345 and http://arxiv.org/abs/1234.5678";
        let urls = find_arxiv_urls(text);
        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"https://arxiv.org/abs/2301.12345".to_string()));
        assert!(urls.contains(&"http://arxiv.org/abs/1234.5678".to_string()));
    }
}
