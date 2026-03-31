#!/usr/bin/env python3
"""
AI OS Terminal Browser – Final version with DuckDuckGo Lite, ad filtering,
increased timeouts, and improved AI decision logic.
"""

import os
import sys
import re
import json
import subprocess
import requests
from urllib.parse import urljoin, urlparse
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
import ollama

# ------------------- Configuration -------------------
MODEL = None
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"

# Trusted folders for file operations (sandbox)
TRUSTED_DIRS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
    os.getcwd(),
]

# Memory file
MEMORY_FILE = os.path.expanduser("~/.ai_os_memory.json")

# Session
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
console = Console()

# ------------------- Model Management -------------------
def get_available_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def pull_model(model_name):
    console.print(f"[yellow]Pulling model {model_name}...[/yellow]")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        console.print(f"[green]Model {model_name} pulled successfully.[/green]")
        return True
    except subprocess.CalledProcessError:
        console.print(f"[red]Failed to pull model {model_name}.[/red]")
        return False

def select_model():
    console.print(Panel("[bold cyan]Model Selection[/bold cyan]", style="bold"))
    models = get_available_models()
    has_table = bool(models)

    if has_table:
        table = Table(title="Available Models")
        table.add_column("#", style="cyan")
        table.add_column("Model Name", style="green")
        for idx, m in enumerate(models, 1):
            table.add_row(str(idx), m)
        console.print(table)
        console.print("\nYou can:")
        console.print("  • Enter a [green]number[/green] to select a model from the table above")
        console.print("  • Type a [green]model name[/green] to pull it (e.g., llama3.2, qwen2.5:7b)")
        console.print("  • Choose one of the options below")
    else:
        console.print("[yellow]No models found. You can pull a new model.[/yellow]")

    while True:
        if has_table:
            prompt_text = "Enter your choice (number, model name, or [Choose from list/Pull new model/Use default]): "
        else:
            prompt_text = "Enter your choice (model name, or [Pull new model/Use default]): "

        choice = Prompt.ask(prompt_text)

        if has_table and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                console.print(f"[bold green]Selected model: {model}[/bold green]")
                if Confirm.ask(f"Update '{model}' to the latest version?", default=False):
                    pull_model(model)
                return model
            else:
                console.print("[red]Invalid number.[/red]")
                continue

        choice_lower = choice.lower()
        if any(phrase in choice_lower for phrase in ["choose", "list", "table"]):
            if has_table:
                continue
            else:
                console.print("[red]No models available. Please pull one.[/red]")
        elif any(phrase in choice_lower for phrase in ["pull", "new model"]):
            new_model = Prompt.ask("Enter model name (e.g., llama3.2, qwen2.5:7b)")
            if new_model and pull_model(new_model):
                return new_model
            else:
                console.print("[red]Pull failed or cancelled. Using default model.[/red]")
                return "llama3.2"
        elif any(phrase in choice_lower for phrase in ["default", "llama3.2"]):
            return "llama3.2"
        else:
            model_name = choice.strip()
            existing = next((m for m in models if m.lower() == model_name.lower()), None)
            if existing:
                model = existing
                console.print(f"[bold green]Using existing model: {model}[/bold green]")
                if Confirm.ask(f"Update '{model}' to the latest version?", default=False):
                    pull_model(model)
                return model
            else:
                console.print(f"[yellow]Model '{model_name}' not found. Attempting to pull...[/yellow]")
                if pull_model(model_name):
                    return model_name
                else:
                    console.print("[red]Pull failed. Using default.[/red]")
                    return "llama3.2"

# ------------------- Memory Management -------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(memory):
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        console.print(f"[red]Failed to save memory: {e}[/red]")

def memory_save(key, value):
    mem = load_memory()
    mem[key] = value
    save_memory(mem)
    return f"Saved memory: {key} = {value}"

def memory_recall(key):
    mem = load_memory()
    value = mem.get(key)
    if value is None:
        return f"No memory found for key: {key}"
    return value

def memory_list():
    mem = load_memory()
    if not mem:
        return "No memories stored."
    return "\n".join(f"- {k}: {v}" for k, v in mem.items())

# ------------------- File System (Sandboxed) -------------------
def _sanitize_path(path):
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        for trusted in TRUSTED_DIRS:
            trusted_abs = os.path.abspath(os.path.expanduser(trusted))
            if abs_path.startswith(trusted_abs + os.sep) or abs_path == trusted_abs:
                return abs_path
        raise PermissionError(f"Access denied: {path} is outside trusted directories.")
    except Exception as e:
        raise PermissionError(f"Invalid path: {e}")

def read_file(path):
    try:
        safe_path = _sanitize_path(path)
        with open(safe_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path, content, append=False):
    try:
        safe_path = _sanitize_path(path)
        mode = 'a' if append else 'w'
        with open(safe_path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"File {'appended to' if append else 'written'}: {safe_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_dir(path="."):
    try:
        safe_path = _sanitize_path(path)
        return "\n".join(os.listdir(safe_path))
    except Exception as e:
        return f"Error listing directory: {e}"

def search_files(pattern, root="."):
    try:
        safe_root = _sanitize_path(root)
        matches = []
        for root_dir, dirs, files in os.walk(safe_root):
            for file in files:
                if pattern in file:
                    matches.append(os.path.join(root_dir, file))
        return "\n".join(matches) if matches else "No matching files found."
    except Exception as e:
        return f"Error searching: {e}"

# ------------------- Shell Commands (with confirmation) -------------------
def run_shell_command(cmd):
    console.print(Panel(f"[yellow]Shell command: {cmd}[/yellow]", title="⚠️ Shell Execution", border_style="red"))
    if not Confirm.ask("Execute this command?", default=False):
        return "Command cancelled by user."
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

# ------------------- Web Helpers (with increased timeouts & ad filtering) -------------------
def fetch_page(url):
    """Fetch a page and return text. Prefers lynx, falls back to requests."""
    try:
        proc = subprocess.run(["lynx", "-dump", "-nolist", url],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            return proc.stdout
        else:
            console.print("[yellow]lynx returned an error. Falling back to requests...[/yellow]")
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")
    except FileNotFoundError:
        console.print("[yellow]lynx not found. Using requests (may be slower).[/yellow]")
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")
        except Exception as e:
            return f"Error fetching page: {e}"
    except Exception as e:
        return f"Error fetching page: {e}"

def extract_links(page_text, base_url):
    try:
        resp = session.get(base_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            text = a.get_text(strip=True)
            links.append((text if text else full_url, full_url))
        return links
    except Exception as e:
        console.print(f"[red]Error extracting links: {e}[/red]")
        return []

def resolve_redirect(url):
    try:
        resp = session.get(url, allow_redirects=False, timeout=15)
        if resp.status_code in (301, 302) and "Location" in resp.headers:
            return resp.headers["Location"]
    except:
        pass
    return url

def search_duckduckgo_lite(query):
    """Perform a DuckDuckGo Lite search, filter out ads, and return formatted text + links."""
    url = "https://lite.duckduckgo.com/lite/"
    params = {"q": query}
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        tables = soup.find_all("table")
        result_table = None
        for tbl in tables:
            if tbl.find("a", href=True):
                result_table = tbl
                break
        if not result_table:
            return "No results found."

        rows = result_table.find_all("tr")
        for row in rows:
            # Look for a cell with a link
            link_cell = row.find("a")
            if link_cell and link_cell.get("href"):
                title = link_cell.get_text(strip=True)
                href = link_cell["href"]
                # Check if this looks like an ad (ad-related keywords)
                if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                    continue
                # Also filter out any href that contains 'y.js' or 'ad_domain' (common ad patterns)
                if "y.js" in href or "ad_domain" in href:
                    continue
                href = resolve_redirect(href)
                results.append((title, href))

        if not results:
            return "No results found."

        result_text = f"Search results for: {query}\n\n"
        for i, (title, href) in enumerate(results[:10], 1):
            result_text += f"{i}. {title}\n   {href}\n\n"
        return result_text
    except Exception as e:
        return f"Search failed: {e}"

# ------------------- AI Decision (Improved Prompt) -------------------
def ask_llm(prompt, context=""):
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": full_prompt}])
        return response["message"]["content"]
    except Exception as e:
        console.print(f"[red]Error calling Ollama: {e}[/red]")
        return None

def decide_next_action(user_query, current_url, page_text, available_links):
    links_text = "\n".join([f"- {text}: {url}" for text, url in available_links[:20]])
    tools_description = """
You have access to additional tools:
- **File system**: read_file(path), write_file(path, content, append=False), list_dir(path="."), search_files(pattern, root=".")
- **Shell**: shell(command) – executes a system command (requires confirmation)
- **Memory**: save_memory(key, value), recall_memory(key), list_memory()
- **Web browsing**: search(query), visit_link(url), extract(answer)

When you need to use a tool, respond with a JSON object that includes "action": "tool", "tool_name": <name>, and "tool_args": <arguments>.
For example:
{"action": "tool", "tool_name": "write_file", "tool_args": {"path": "~/Documents/note.txt", "content": "Hello"}}
{"action": "tool", "tool_name": "shell", "tool_args": {"command": "ls -la"}}
{"action": "tool", "tool_name": "save_memory", "tool_args": {"key": "important", "value": "some value"}}
{"action": "tool", "tool_name": "recall_memory", "tool_args": {"key": "important"}}

If you want to continue web browsing, use the previous actions (search, visit_link, extract, stop).
"""

    prompt = f"""
You are an AI OS Terminal Browser. The user's goal is: "{user_query}"
You are currently on: {current_url}

Page content (first 2000 chars):
{page_text[:2000]}

Available links on this page:
{links_text}

{tools_description}

**IMPORTANT INSTRUCTIONS:**
- If the current page is a **search results page** (like DuckDuckGo or Google results), the goal is **NOT yet achieved**. You must choose a relevant link to visit and then extract the actual information from that page.
- Only **stop** if the page already contains the exact answer the user wants, or if you have already extracted it.
- If you are on a search results page, use "visit_link" with one of the URLs from the list.
- If you are on a news article or a page that likely contains the answer, use "extract" to pull out the relevant information.
- Only use "search" again if the current page is completely irrelevant.

Decide what to do next. Choose one action from:
- "search" (with query) – if you need a different search.
- "visit_link" (with url) – to go to a specific page from the links above.
- "extract" (with answer) – if the current page contains the answer.
- "tool" – to use filesystem, shell, or memory.
- "stop" – only when the goal is fully satisfied.

Respond with a JSON object. Example responses:
{{"action": "visit_link", "url": "https://techcrunch.com/ai/", "reason": "This looks like a major AI news source"}}
{{"action": "extract", "answer": "The latest AI news: ...", "reason": "Found the answer on this page"}}
{{"action": "search", "query": "latest AI news 2026", "reason": "No relevant results found"}}
{{"action": "tool", "tool_name": "write_file", "tool_args": {{"path": "~/Documents/notes.txt", "content": "Meeting notes"}}, "reason": "User asked to save notes"}}
{{"action": "stop", "reason": "Goal achieved"}}

Do NOT stop on a search results page. Always try to get to the actual content.
"""
    response = ask_llm(prompt)
    if not response:
        return {"action": "stop", "reason": "LLM error"}
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"action": "stop", "reason": "Could not parse LLM response"}
    except json.JSONDecodeError:
        return {"action": "stop", "reason": "Invalid JSON from LLM"}

# ------------------- Main Loop -------------------
def main():
    global MODEL
    console.print(Panel.fit("[bold cyan]AI OS Terminal Browser[/bold cyan] - AI-powered assistant with web, file, shell, and memory", style="bold"))

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Ollama not found. Please install Ollama and make sure it's running.[/red]")
        sys.exit(1)

    MODEL = select_model()

    console.print("\n[bold yellow]Enter your goal. You can ask me to browse the web, read/write files, run commands, or remember things.[/bold yellow]")
    console.print("Type 'exit' to quit. During session, type '!model' to switch models, '!memory' to view memory.\n")

    user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
    if user_query.lower() in ("exit", "quit"):
        return

    current_url = None
    page_text = ""
    available_links = []

    while True:
        # Special commands
        if user_query.strip() == "!model":
            MODEL = select_model()
            user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
            if user_query.lower() in ("exit", "quit"):
                break
            continue
        elif user_query.strip() == "!memory":
            mem_list = memory_list()
            console.print(Panel(mem_list, title="Memory Contents", border_style="green"))
            user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
            if user_query.lower() in ("exit", "quit"):
                break
            continue

        # Perform search or fetch page if web browsing
        if current_url is None and not user_query.startswith("!"):
            console.print(f"[green]Searching for: {user_query}[/green]")
            page_text = search_duckduckgo_lite(user_query)

            # Extract links from search results (again, to get the same filtered list)
            url = "https://lite.duckduckgo.com/lite/"
            params = {"q": user_query}
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                tables = soup.find_all("table")
                result_table = None
                for tbl in tables:
                    if tbl.find("a", href=True):
                        result_table = tbl
                        break
                if result_table:
                    available_links = []
                    rows = result_table.find_all("tr")
                    for row in rows:
                        link_cell = row.find("a")
                        if link_cell and link_cell.get("href"):
                            title = link_cell.get_text(strip=True)
                            href = link_cell["href"]
                            # Filter ads same as above
                            if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                                continue
                            if "y.js" in href or "ad_domain" in href:
                                continue
                            href = resolve_redirect(href)
                            available_links.append((title, href))
            except Exception as e:
                console.print(f"[red]Error extracting links: {e}[/red]")
                available_links = []
        elif current_url is not None:
            console.print(f"[green]Fetching: {current_url}[/green]")
            page_text = fetch_page(current_url)
            console.print("[dim]Extracting links from page...[/dim]")
            available_links = extract_links(page_text, current_url)

        # Show preview if we have page content
        if page_text and page_text != "No results found." and not user_query.startswith("!"):
            preview = page_text[:1000] + "..." if len(page_text) > 1000 else page_text
            console.print(Panel(preview, title=f"Current page: {current_url if current_url else 'Search Results'}", border_style="blue"))

        # Ask AI for next action
        console.print("[yellow]AI is thinking...[/yellow]")
        decision = decide_next_action(user_query, current_url or "Search Results page", page_text, available_links)
        console.print(f"[bold]AI decision:[/bold] {decision.get('reason', 'No reason')}")

        action = decision.get("action")
        if action == "stop":
            console.print("[green]Done![/green]")
            if decision.get("answer"):
                console.print(Markdown(decision["answer"]))
            break
        elif action == "search":
            new_query = decision.get("query")
            if not new_query:
                console.print("[red]No search query provided. Exiting.[/red]")
                break
            console.print(f"[blue]Searching for: {new_query}[/blue]")
            current_url = None
            user_query = new_query
        elif action == "visit_link":
            url = decision.get("url")
            if not url:
                console.print("[red]No URL provided. Exiting.[/red]")
                break
            url = resolve_redirect(url)
            if current_url and not urlparse(url).netloc:
                url = urljoin(current_url, url)
            console.print(f"[blue]Following link: {url}[/blue]")
            current_url = url
        elif action == "extract":
            answer = decision.get("answer")
            if answer:
                console.print(Panel(Markdown(answer), title="Extracted Answer", border_style="green"))
            else:
                console.print("[red]No answer extracted. Exiting.[/red]")
            break
        elif action == "tool":
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args", {})
            if not tool_name:
                console.print("[red]No tool name provided.[/red]")
                break
            # Execute the tool
            if tool_name == "read_file":
                path = tool_args.get("path")
                if path:
                    result = read_file(path)
                    console.print(Panel(result, title=f"File: {path}", border_style="cyan"))
                else:
                    console.print("[red]Missing path for read_file[/red]")
            elif tool_name == "write_file":
                path = tool_args.get("path")
                content = tool_args.get("content", "")
                append = tool_args.get("append", False)
                if path:
                    result = write_file(path, content, append)
                    console.print(Panel(result, title="Write Result", border_style="green"))
                else:
                    console.print("[red]Missing path for write_file[/red]")
            elif tool_name == "list_dir":
                path = tool_args.get("path", ".")
                result = list_dir(path)
                console.print(Panel(result, title=f"Directory: {path}", border_style="cyan"))
            elif tool_name == "search_files":
                pattern = tool_args.get("pattern")
                root = tool_args.get("root", ".")
                if pattern:
                    result = search_files(pattern, root)
                    console.print(Panel(result, title=f"Search for '{pattern}'", border_style="cyan"))
                else:
                    console.print("[red]Missing pattern for search_files[/red]")
            elif tool_name == "shell":
                cmd = tool_args.get("command")
                if cmd:
                    result = run_shell_command(cmd)
                    console.print(Panel(result, title="Shell Output", border_style="yellow"))
                else:
                    console.print("[red]Missing command for shell[/red]")
            elif tool_name == "save_memory":
                key = tool_args.get("key")
                value = tool_args.get("value")
                if key and value:
                    result = memory_save(key, value)
                    console.print(Panel(result, title="Memory Saved", border_style="green"))
                else:
                    console.print("[red]Missing key or value for save_memory[/red]")
            elif tool_name == "recall_memory":
                key = tool_args.get("key")
                if key:
                    result = memory_recall(key)
                    console.print(Panel(result, title=f"Memory: {key}", border_style="cyan"))
                else:
                    console.print("[red]Missing key for recall_memory[/red]")
            elif tool_name == "list_memory":
                result = memory_list()
                console.print(Panel(result, title="Memory List", border_style="cyan"))
            else:
                console.print(f"[red]Unknown tool: {tool_name}[/red]")
        else:
            console.print(f"[red]Unknown action: {action}. Exiting.[/red]")
            break

        # User interaction
        if not Confirm.ask("[bold yellow]Continue with AI?[/bold yellow]", default=True):
            manual = Prompt.ask("Enter command (URL, !model, !memory, 'exit')")
            if manual.lower() == "exit":
                break
            elif manual.strip() == "!model":
                MODEL = select_model()
                user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
            elif manual.strip() == "!memory":
                mem_list = memory_list()
                console.print(Panel(mem_list, title="Memory Contents", border_style="green"))
                user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
            else:
                current_url = manual
                user_query = ""
        else:
            pass

    console.print("[bold green]Goodbye![/bold green]")

if __name__ == "__main__":
    main()
