
#!/usr/bin/env python3
"""
AI OS Terminal Browser – Full Version
Features: Web browsing, file operations, shell, memory, Git, project awareness,
multi-file editing, summarization, comparison, scheduling, and chatbot API.
"""

import os
import sys
import re
import json
import subprocess
import requests
import difflib
import threading
import time
import argparse
import ast                     # for project indexing
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

# ---------- Optional imports ----------
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    print("Warning: 'schedule' not installed. Scheduled tasks disabled.")
    print("Install with: pip install schedule")

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

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
CONVERSATION_FILE = os.path.expanduser("~/.ai_os_conversation.json")

# Session
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
console = Console()

# ------------------- Global objects -------------------
project_index = None            # will be built lazily
scheduler_thread = None
scheduler_running = False

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

# ------------------- Conversation Memory -------------------
def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_conversation(conv):
    try:
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conv, f, indent=2)
    except Exception as e:
        console.print(f"[red]Failed to save conversation: {e}[/red]")

def add_to_conversation(role, content):
    conv = load_conversation()
    conv.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    save_conversation(conv)

def get_conversation_context(limit=10):
    conv = load_conversation()
    return conv[-limit:] if limit else conv

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

# ------------------- Codebase Awareness (Project Index) -------------------
class ProjectIndex:
    def __init__(self, root_path):
        self.root = Path(root_path).resolve()
        self.files = []          # relative paths
        self.imports = {}        # rel_path -> list of imported modules
        self.definitions = {}    # symbol -> list of (rel_path, line)

    def build(self):
        for file_path in self.root.rglob("*"):
            if file_path.suffix in ('.py', '.js', '.ts', '.go', '.java', '.c', '.cpp'):
                rel = str(file_path.relative_to(self.root))
                self.files.append(rel)
                if file_path.suffix == '.py':
                    self._parse_python(file_path, rel)

    def _parse_python(self, file_path, rel):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
        except (SyntaxError, Exception):
            return
        # Collect imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        self.imports[rel] = imports
        # Collect definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                self.definitions.setdefault(name, []).append((rel, node.lineno))

    def get_summary(self):
        lines = [f"Project root: {self.root}"]
        lines.append(f"Files: {len(self.files)}")
        top_files = [f for f in self.files if '/' not in f][:10]
        lines.append(f"Top-level files: {', '.join(top_files)}")
        top_symbols = list(self.definitions.keys())[:20]
        lines.append(f"Top symbols: {', '.join(top_symbols)}")
        return "\n".join(lines)

    def find_definition(self, symbol):
        return self.definitions.get(symbol, [])

    def find_imports_of(self, module):
        results = []
        for file, imps in self.imports.items():
            if any(module in imp for imp in imps):
                results.append(file)
        return results

def get_project_index(root="."):
    global project_index
    safe_root = _sanitize_path(root)
    if project_index is None or project_index.root != Path(safe_root).resolve():
        project_index = ProjectIndex(safe_root)
        project_index.build()
    return project_index

def project_info(root="."):
    index = get_project_index(root)
    return index.get_summary()

# ------------------- Git Integration -------------------
def git_status(repo_path="."):
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", safe_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return result.stdout if result.stdout.strip() else "Working tree clean"
    except Exception as e:
        return f"Error: {e}"

def git_diff(repo_path=".", staged=False):
    safe_path = _sanitize_path(repo_path)
    cmd = ["git", "-C", safe_path, "diff"]
    if staged:
        cmd.append("--staged")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout if result.stdout.strip() else "No changes to show"
    except Exception as e:
        return f"Error: {e}"

def git_commit(repo_path=".", message=""):
    if not message:
        return "Commit message required"
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", safe_path, "commit", "-m", message],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Commit failed: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def git_branch(repo_path=".", new_branch=None):
    safe_path = _sanitize_path(repo_path)
    if new_branch:
        cmd = ["git", "-C", safe_path, "checkout", "-b", new_branch]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() or result.stderr
        except Exception as e:
            return f"Error: {e}"
    else:
        try:
            result = subprocess.run(
                ["git", "-C", safe_path, "branch", "--show-current"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

def git_log(repo_path=".", n=10):
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", safe_path, "log", f"-{n}", "--oneline"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.stdout.strip() else "No commits"
    except Exception as e:
        return f"Error: {e}"

# ------------------- Summarization -------------------
def summarize_text(text, max_length=500):
    """Summarize text using the LLM."""
    prompt = f"Summarize the following text concisely (max {max_length} words):\n\n{text[:8000]}"
    response = ask_llm(prompt)
    return response or "Summarization failed."

# ------------------- Comparison Tools -------------------
def compare_files(file1, file2):
    try:
        content1 = read_file(file1).splitlines()
        content2 = read_file(file2).splitlines()
    except Exception as e:
        return f"Error reading files: {e}"
    diff = difflib.unified_diff(content1, content2, fromfile=file1, tofile=file2)
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Files are identical."

def compare_webpages(url1, url2):
    text1 = fetch_page(url1)
    text2 = fetch_page(url2)
    # Simple diff on text (may be large; we can truncate)
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    diff = difflib.unified_diff(lines1, lines2, fromfile=url1, tofile=url2)
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Pages are identical."

# ------------------- Scheduled Tasks -------------------
scheduled_jobs = []  # list of dicts with id, command, schedule string

def start_scheduler():
    """Start background scheduler thread."""
    global scheduler_running, scheduler_thread
    if scheduler_running or not HAS_SCHEDULE:
        return
    scheduler_running = True

    def run_loop():
        while scheduler_running:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_loop, daemon=True)
    scheduler_thread.start()

def stop_scheduler():
    global scheduler_running
    scheduler_running = False
    if scheduler_thread and scheduler_thread.is_alive():
        scheduler_thread.join(timeout=2)

def _execute_scheduled_task(job):
    """Run a scheduled task (callback for schedule)."""
    command = job['command']
    tool_args = job.get('tool_args', {})
    if command.startswith('shell:'):
        cmd = command[6:]
        result = run_shell_command(cmd, auto_confirm=True)
        console.print(f"[yellow]Scheduled task executed: {cmd}[/yellow]\n{result}")
    elif command.startswith('fetch:'):
        url = command[6:]
        content = fetch_page(url)
        output = tool_args.get('output', 'schedule_output.txt')
        write_file(output, content, append=True)
        console.print(f"[yellow]Scheduled fetch saved to {output}[/yellow]")
    else:
        console.print(f"[red]Unknown scheduled command: {command}[/red]")

def schedule_task(command, time_str, tool_args=None):
    """
    Schedule a task.
    time_str: e.g., '09:00' (daily at that time) or 'in 5 minutes' (not supported yet)
    """
    if not HAS_SCHEDULE:
        return "Schedule library not installed. Install with: pip install schedule"
    start_scheduler()
    job = {
        'command': command,
        'tool_args': tool_args or {}
    }
    try:
        # Simple: daily at time
        schedule.every().day.at(time_str).do(_execute_scheduled_task, job)
        return f"Scheduled {command} at {time_str} daily"
    except Exception as e:
        return f"Failed to schedule: {e}"

def list_scheduled_tasks():
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if not jobs:
        return "No scheduled tasks."
    lines = []
    for i, job in enumerate(jobs, 1):
        lines.append(f"{i}. {job}")  # schedule.job has a nice __str__
    return "\n".join(lines)

def cancel_scheduled_task(index):
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if 1 <= index <= len(jobs):
        schedule.cancel_job(jobs[index-1])
        return f"Cancelled task {index}"
    else:
        return "Invalid task index"

# ------------------- Shell Commands -------------------
def run_shell_command(cmd, auto_confirm=False):
    if not auto_confirm:
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

# ------------------- Web Helpers -------------------
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
            link_cell = row.find("a")
            if link_cell and link_cell.get("href"):
                title = link_cell.get_text(strip=True)
                href = link_cell["href"]
                if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                    continue
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

# ------------------- AI Decision -------------------
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
- **Git**: git_status(repo_path="."), git_diff(repo_path=".", staged=False), git_commit(repo_path=".", message=""), git_branch(repo_path=".", new_branch=None), git_log(repo_path=".", n=10)
- **Project**: project_info(root=".") – gives overview of codebase
- **Summarization**: summarize(text) – returns concise summary
- **Comparison**: compare_files(file1, file2), compare_webpages(url1, url2)
- **Scheduling**: schedule_task(command, time_str, tool_args=None), list_scheduled_tasks(), cancel_scheduled_task(index)
- **Web browsing**: search(query), visit_link(url), extract(answer)

When you need to use a tool, respond with a JSON object that includes "action": "tool", "tool_name": <name>, and "tool_args": <arguments>.
For example:
{{"action": "tool", "tool_name": "write_file", "tool_args": {{"path": "~/Documents/note.txt", "content": "Hello"}}}}
{{"action": "tool", "tool_name": "shell", "tool_args": {{"command": "ls -la"}}}}
{{"action": "tool", "tool_name": "save_memory", "tool_args": {{"key": "important", "value": "some value"}}}}
{{"action": "tool", "tool_name": "git_status", "tool_args": {{"repo_path": "."}}}}
{{"action": "tool", "tool_name": "summarize", "tool_args": {{"text": "long text here"}}}}
{{"action": "tool", "tool_name": "compare_files", "tool_args": {{"file1": "a.txt", "file2": "b.txt"}}}}
{{"action": "tool", "tool_name": "schedule_task", "tool_args": {{"command": "shell:echo hello", "time_str": "09:00"}}}}

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
- "tool" – to use filesystem, shell, memory, Git, project, summarization, comparison, scheduling.
- "multi_edit" – to edit multiple files at once: {{"action": "multi_edit", "edits": [{{"path": "...", "content": "...", "append": false}}]}}
- "stop" – only when the goal is fully satisfied.

Respond with a JSON object. Example responses:
{{"action": "visit_link", "url": "https://techcrunch.com/ai/", "reason": "This looks like a major AI news source"}}
{{"action": "extract", "answer": "The latest AI news: ...", "reason": "Found the answer on this page"}}
{{"action": "search", "query": "latest AI news 2026", "reason": "No relevant results found"}}
{{"action": "tool", "tool_name": "write_file", "tool_args": {{"path": "~/Documents/notes.txt", "content": "Meeting notes"}}, "reason": "User asked to save notes"}}
{{"action": "multi_edit", "edits": [{{"path": "main.py", "content": "print('hello')"}}, {{"path": "utils.py", "content": "def foo(): pass"}}], "reason": "Add new functions"}}
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

# ------------------- Autonomous Query Processing -------------------
def process_query(user_query, interactive=False):
    """
    Process a single query autonomously, returning the final answer.
    If interactive is True, it will show output and ask for confirmation at each step.
    Otherwise, it runs silently and returns the final extracted answer.
    """
    current_url = None
    page_text = ""
    available_links = []
    final_answer = None

    while True:
        # Special commands
        if user_query.strip() == "!model":
            # Not used in non-interactive mode
            break
        elif user_query.strip() == "!memory":
            # Not used
            break

        # Perform search or fetch page if web browsing
        if current_url is None and not user_query.startswith("!"):
            if interactive:
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
                            if any(phrase in title.lower() for phrase in ["ad", "sponsored", "promoted"]):
                                continue
                            if "y.js" in href or "ad_domain" in href:
                                continue
                            href = resolve_redirect(href)
                            available_links.append((title, href))
            except Exception as e:
                if interactive:
                    console.print(f"[red]Error extracting links: {e}[/red]")
                available_links = []
        elif current_url is not None:
            if interactive:
                console.print(f"[green]Fetching: {current_url}[/green]")
            page_text = fetch_page(current_url)
            if interactive:
                console.print("[dim]Extracting links from page...[/dim]")
            available_links = extract_links(page_text, current_url)

        # Show preview if interactive
        if interactive and page_text and page_text != "No results found." and not user_query.startswith("!"):
            preview = page_text[:1000] + "..." if len(page_text) > 1000 else page_text
            console.print(Panel(preview, title=f"Current page: {current_url if current_url else 'Search Results'}", border_style="blue"))

        # Ask AI for next action
        if interactive:
            console.print("[yellow]AI is thinking...[/yellow]")
        decision = decide_next_action(user_query, current_url or "Search Results page", page_text, available_links)
        if interactive:
            console.print(f"[bold]AI decision:[/bold] {decision.get('reason', 'No reason')}")

        action = decision.get("action")
        if action == "stop":
            final_answer = decision.get("answer")
            if interactive:
                console.print("[green]Done![/green]")
                if final_answer:
                    console.print(Markdown(final_answer))
            break
        elif action == "search":
            new_query = decision.get("query")
            if not new_query:
                if interactive:
                    console.print("[red]No search query provided. Exiting.[/red]")
                break
            if interactive:
                console.print(f"[blue]Searching for: {new_query}[/blue]")
            current_url = None
            user_query = new_query
        elif action == "visit_link":
            url = decision.get("url")
            if not url:
                if interactive:
                    console.print("[red]No URL provided. Exiting.[/red]")
                break
            url = resolve_redirect(url)
            if current_url and not urlparse(url).netloc:
                url = urljoin(current_url, url)
            if interactive:
                console.print(f"[blue]Following link: {url}[/blue]")
            current_url = url
        elif action == "extract":
            answer = decision.get("answer")
            if answer:
                final_answer = answer
                if interactive:
                    console.print(Panel(Markdown(answer), title="Extracted Answer", border_style="green"))
            else:
                if interactive:
                    console.print("[red]No answer extracted. Exiting.[/red]")
            break
        elif action == "multi_edit":
            edits = decision.get("edits", [])
            for edit in edits:
                path = edit.get("path")
                content = edit.get("content", "")
                append = edit.get("append", False)
                if path:
                    result = write_file(path, content, append)
                    if interactive:
                        console.print(Panel(result, title=f"Write {path}", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing path in edit[/red]")
            # After multi-edit, we may want to stop or continue? Let's continue to next decision.
        elif action == "tool":
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args", {})
            if not tool_name:
                if interactive:
                    console.print("[red]No tool name provided.[/red]")
                break
            # Execute the tool
            if tool_name == "read_file":
                path = tool_args.get("path")
                if path:
                    result = read_file(path)
                    if interactive:
                        console.print(Panel(result, title=f"File: {path}", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing path for read_file[/red]")
            elif tool_name == "write_file":
                path = tool_args.get("path")
                content = tool_args.get("content", "")
                append = tool_args.get("append", False)
                if path:
                    result = write_file(path, content, append)
                    if interactive:
                        console.print(Panel(result, title="Write Result", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing path for write_file[/red]")
            elif tool_name == "list_dir":
                path = tool_args.get("path", ".")
                result = list_dir(path)
                if interactive:
                    console.print(Panel(result, title=f"Directory: {path}", border_style="cyan"))
            elif tool_name == "search_files":
                pattern = tool_args.get("pattern")
                root = tool_args.get("root", ".")
                if pattern:
                    result = search_files(pattern, root)
                    if interactive:
                        console.print(Panel(result, title=f"Search for '{pattern}'", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing pattern for search_files[/red]")
            elif tool_name == "shell":
                cmd = tool_args.get("command")
                if cmd:
                    result = run_shell_command(cmd, auto_confirm=not interactive)  # auto-confirm in non-interactive
                    if interactive:
                        console.print(Panel(result, title="Shell Output", border_style="yellow"))
                else:
                    if interactive:
                        console.print("[red]Missing command for shell[/red]")
            elif tool_name == "save_memory":
                key = tool_args.get("key")
                value = tool_args.get("value")
                if key and value:
                    result = memory_save(key, value)
                    if interactive:
                        console.print(Panel(result, title="Memory Saved", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing key or value for save_memory[/red]")
            elif tool_name == "recall_memory":
                key = tool_args.get("key")
                if key:
                    result = memory_recall(key)
                    if interactive:
                        console.print(Panel(result, title=f"Memory: {key}", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing key for recall_memory[/red]")
            elif tool_name == "list_memory":
                result = memory_list()
                if interactive:
                    console.print(Panel(result, title="Memory List", border_style="cyan"))
            elif tool_name == "project_info":
                root = tool_args.get("root", ".")
                result = project_info(root)
                if interactive:
                    console.print(Panel(result, title="Project Info", border_style="green"))
            elif tool_name == "git_status":
                repo = tool_args.get("repo_path", ".")
                result = git_status(repo)
                if interactive:
                    console.print(Panel(result, title="Git Status", border_style="green"))
            elif tool_name == "git_diff":
                repo = tool_args.get("repo_path", ".")
                staged = tool_args.get("staged", False)
                result = git_diff(repo, staged)
                if interactive:
                    console.print(Panel(result, title="Git Diff", border_style="green"))
            elif tool_name == "git_commit":
                repo = tool_args.get("repo_path", ".")
                message = tool_args.get("message", "")
                result = git_commit(repo, message)
                if interactive:
                    console.print(Panel(result, title="Git Commit", border_style="green"))
            elif tool_name == "git_branch":
                repo = tool_args.get("repo_path", ".")
                new_branch = tool_args.get("new_branch", None)
                result = git_branch(repo, new_branch)
                if interactive:
                    console.print(Panel(result, title="Git Branch", border_style="green"))
            elif tool_name == "git_log":
                repo = tool_args.get("repo_path", ".")
                n = tool_args.get("n", 10)
                result = git_log(repo, n)
                if interactive:
                    console.print(Panel(result, title="Git Log", border_style="green"))
            elif tool_name == "summarize":
                text = tool_args.get("text", "")
                if text:
                    result = summarize_text(text)
                    if interactive:
                        console.print(Panel(result, title="Summary", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing text for summarize[/red]")
            elif tool_name == "compare_files":
                file1 = tool_args.get("file1")
                file2 = tool_args.get("file2")
                if file1 and file2:
                    result = compare_files(file1, file2)
                    if interactive:
                        console.print(Panel(result, title="File Comparison", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing file1 or file2 for compare_files[/red]")
            elif tool_name == "compare_webpages":
                url1 = tool_args.get("url1")
                url2 = tool_args.get("url2")
                if url1 and url2:
                    result = compare_webpages(url1, url2)
                    if interactive:
                        console.print(Panel(result, title="Webpage Comparison", border_style="cyan"))
                else:
                    if interactive:
                        console.print("[red]Missing url1 or url2 for compare_webpages[/red]")
            elif tool_name == "schedule_task":
                command = tool_args.get("command")
                time_str = tool_args.get("time_str")
                task_tool_args = tool_args.get("tool_args", {})
                if command and time_str:
                    result = schedule_task(command, time_str, task_tool_args)
                    if interactive:
                        console.print(Panel(result, title="Schedule Task", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing command or time_str for schedule_task[/red]")
            elif tool_name == "list_scheduled_tasks":
                result = list_scheduled_tasks()
                if interactive:
                    console.print(Panel(result, title="Scheduled Tasks", border_style="cyan"))
            elif tool_name == "cancel_scheduled_task":
                index = tool_args.get("index")
                if index is not None:
                    result = cancel_scheduled_task(index)
                    if interactive:
                        console.print(Panel(result, title="Cancel Task", border_style="green"))
                else:
                    if interactive:
                        console.print("[red]Missing index for cancel_scheduled_task[/red]")
            else:
                if interactive:
                    console.print(f"[red]Unknown tool: {tool_name}[/red]")
        else:
            if interactive:
                console.print(f"[red]Unknown action: {action}. Exiting.[/red]")
            break

        # In non-interactive mode, we continue until stop is reached.
        # In interactive, we may ask for user continuation.
        if interactive:
            if not Confirm.ask("[bold yellow]Continue with AI?[/bold yellow]", default=True):
                manual = Prompt.ask("Enter command (URL, !model, !memory, 'exit')")
                if manual.lower() == "exit":
                    break
                elif manual.strip() == "!model":
                    global MODEL
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

    return final_answer

# ------------------- Flask API Server -------------------
def start_api_server(host='127.0.0.1', port=5000):
    if not HAS_FLASK:
        console.print("[red]Flask not installed. Cannot start API server.[/red]")
        return

    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field'}), 400
        query = data['query']
        # Optionally use conversation history
        conv = load_conversation()
        # Process query (non-interactive)
        answer = process_query(query, interactive=False)
        # Store in conversation
        add_to_conversation('user', query)
        add_to_conversation('assistant', answer)
        return jsonify({'response': answer})

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})

    console.print(f"[green]Starting API server on http://{host}:{port}[/green]")
    app.run(host=host, port=port)

# ------------------- Main Interactive Loop -------------------
def interactive_main():
    global MODEL
    console.print(Panel.fit("[bold cyan]AI OS Terminal Browser[/bold cyan] - AI-powered assistant with web, file, shell, Git, scheduling, and memory", style="bold"))

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Ollama not found. Please install Ollama and make sure it's running.[/red]")
        sys.exit(1)

    MODEL = select_model()

    console.print("\n[bold yellow]Enter your goal. You can ask me to browse the web, read/write files, run commands, manage Git, schedule tasks, etc.[/bold yellow]")
    console.print("Type 'exit' to quit. During session, type '!model' to switch models, '!memory' to view memory.\n")

    # Start scheduler thread
    start_scheduler()

    user_query = Prompt.ask("[bold yellow]What do you want to do?[/bold yellow]")
    if user_query.lower() in ("exit", "quit"):
        return

    # Process query interactively
    process_query(user_query, interactive=True)

    # Stop scheduler on exit
    stop_scheduler()
    console.print("[bold green]Goodbye![/bold green]")

# ------------------- Command-line Entry -------------------
def main():
    parser = argparse.ArgumentParser(description="AI OS Terminal Browser")
    parser.add_argument('--serve', action='store_true', help="Start API server instead of interactive mode")
    parser.add_argument('--host', default='127.0.0.1', help="Host for API server (default: 127.0.0.1)")
    parser.add_argument('--port', type=int, default=5000, help="Port for API server (default: 5000)")
    args = parser.parse_args()

    if args.serve:
        start_api_server(host=args.host, port=args.port)
    else:
        interactive_main()

if __name__ == "__main__":
    main()
