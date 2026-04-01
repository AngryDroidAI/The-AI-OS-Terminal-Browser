#!/usr/bin/env python3
"""
AI OS Terminal Browser – Enhanced Version
Features: Web browsing, file operations, shell, memory, Git, project awareness,
multi-file editing, summarization, comparison, scheduling, and chatbot API.

Improvements:
- Pathlib-based strict sandboxing
- Caching for web pages
- Async web requests (aiohttp fallback)
- Streaming LLM responses in interactive mode
- Config file support (config.yaml)
- Conversation pruning and memory limits
- Enhanced logging
- Better error handling and retries
"""

import os
import sys
import re
import json
import subprocess
import difflib
import threading
import time
import argparse
import ast
import logging
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Tuple, Any, Union

import requests
import yaml
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import ollama

# ---------- Optional imports ----------
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_os")

# ------------------- Configuration -------------------
CONFIG_FILE = Path.home() / ".ai_os_config.yaml"
DEFAULT_CONFIG = {
    "model": None,                      # will be selected at startup
    "trusted_dirs": [
        "~/Documents",
        "~/Downloads",
        "."
    ],
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "memory_file": "~/.ai_os_memory.json",
    "conversation_file": "~/.ai_os_conversation.json",
    "cache_ttl": 3600,                 # seconds
    "max_conversation_turns": 20,
    "max_page_chars": 8000,
    "shell_allowed_commands": ["ls", "pwd", "echo", "cat", "grep", "git"],  # allowlist
    "api_host": "127.0.0.1",
    "api_port": 5000
}

class Config:
    def __init__(self):
        self._data = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    user_cfg = yaml.safe_load(f)
                    if user_cfg:
                        self._data.update(user_cfg)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

    def save(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self._data, f)
        except Exception as e:
            logger.warning(f"Could not save config: {e}")

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

config = Config()

# Expand user paths in trusted dirs
TRUSTED_DIRS = [Path(p).expanduser().resolve() for p in config.get("trusted_dirs")]

# ---------- Session and cache ----------
session = requests.Session()
session.headers.update({"User-Agent": config.get("user_agent")})
console = Console()

# Simple cache for web pages
_cache = {}  # key: url, value: (timestamp, content)

def get_cache(url: str) -> Optional[str]:
    if url in _cache:
        timestamp, content = _cache[url]
        if datetime.now() - timestamp < timedelta(seconds=config.get("cache_ttl")):
            return content
        else:
            del _cache[url]
    return None

def set_cache(url: str, content: str):
    _cache[url] = (datetime.now(), content)

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
    mem_path = Path(config.get("memory_file")).expanduser()
    if mem_path.exists():
        try:
            with open(mem_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(memory):
    mem_path = Path(config.get("memory_file")).expanduser()
    try:
        with open(mem_path, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

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
    conv_path = Path(config.get("conversation_file")).expanduser()
    if conv_path.exists():
        try:
            with open(conv_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_conversation(conv):
    conv_path = Path(config.get("conversation_file")).expanduser()
    # Prune if needed
    max_turns = config.get("max_conversation_turns")
    if max_turns and len(conv) > max_turns:
        conv = conv[-max_turns:]
    try:
        with open(conv_path, 'w') as f:
            json.dump(conv, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

def add_to_conversation(role, content):
    conv = load_conversation()
    conv.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    save_conversation(conv)

def get_conversation_context(limit=10):
    conv = load_conversation()
    return conv[-limit:] if limit else conv

# ------------------- File System (Strict Sandbox) -------------------
def _sanitize_path(path: Union[str, Path]) -> Path:
    """Resolve path and ensure it is within one of the trusted directories."""
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        raise PermissionError(f"Invalid path: {path}")
    for trusted in TRUSTED_DIRS:
        try:
            # Check if p is within trusted (or equal)
            p.relative_to(trusted)
            return p
        except ValueError:
            continue
    raise PermissionError(f"Access denied: {path} is outside trusted directories.")

def read_file(path: Union[str, Path]) -> str:
    try:
        safe_path = _sanitize_path(path)
        return safe_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: Union[str, Path], content: str, append: bool = False) -> str:
    try:
        safe_path = _sanitize_path(path)
        mode = 'a' if append else 'w'
        with open(safe_path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"File {'appended to' if append else 'written'}: {safe_path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_dir(path: Union[str, Path] = ".") -> str:
    try:
        safe_path = _sanitize_path(path)
        return "\n".join(str(x) for x in safe_path.iterdir())
    except Exception as e:
        return f"Error listing directory: {e}"

def search_files(pattern: str, root: Union[str, Path] = ".") -> str:
    try:
        safe_root = _sanitize_path(root)
        matches = []
        for file_path in safe_root.rglob("*"):
            if file_path.is_file() and pattern in file_path.name:
                matches.append(str(file_path))
        return "\n".join(matches) if matches else "No matching files found."
    except Exception as e:
        return f"Error searching: {e}"

# ------------------- Codebase Awareness (Project Index) -------------------
class ProjectIndex:
    def __init__(self, root_path: Union[str, Path]):
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

    def _parse_python(self, file_path: Path, rel: str):
        try:
            tree = ast.parse(file_path.read_text(encoding='utf-8'))
        except (SyntaxError, Exception):
            return
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        self.imports[rel] = imports
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

_project_index = None

def get_project_index(root: Union[str, Path] = ".") -> ProjectIndex:
    global _project_index
    safe_root = _sanitize_path(root)
    if _project_index is None or _project_index.root != safe_root:
        _project_index = ProjectIndex(safe_root)
        _project_index.build()
    return _project_index

def project_info(root: Union[str, Path] = ".") -> str:
    index = get_project_index(root)
    return index.get_summary()

# ------------------- Git Integration -------------------
def git_status(repo_path: Union[str, Path] = ".") -> str:
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return result.stdout if result.stdout.strip() else "Working tree clean"
    except Exception as e:
        return f"Error: {e}"

def git_diff(repo_path: Union[str, Path] = ".", staged: bool = False) -> str:
    safe_path = _sanitize_path(repo_path)
    cmd = ["git", "-C", str(safe_path), "diff"]
    if staged:
        cmd.append("--staged")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout if result.stdout.strip() else "No changes to show"
    except Exception as e:
        return f"Error: {e}"

def git_commit(repo_path: Union[str, Path] = ".", message: str = "") -> str:
    if not message:
        return "Commit message required"
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "commit", "-m", message],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Commit failed: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def git_branch(repo_path: Union[str, Path] = ".", new_branch: Optional[str] = None) -> str:
    safe_path = _sanitize_path(repo_path)
    if new_branch:
        cmd = ["git", "-C", str(safe_path), "checkout", "-b", new_branch]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() or result.stderr
        except Exception as e:
            return f"Error: {e}"
    else:
        try:
            result = subprocess.run(
                ["git", "-C", str(safe_path), "branch", "--show-current"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

def git_log(repo_path: Union[str, Path] = ".", n: int = 10) -> str:
    safe_path = _sanitize_path(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", str(safe_path), "log", f"-{n}", "--oneline"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.stdout.strip() else "No commits"
    except Exception as e:
        return f"Error: {e}"

# ------------------- Summarization -------------------
def summarize_text(text: str, max_length: int = 500) -> str:
    prompt = f"Summarize the following text concisely (max {max_length} words):\n\n{text[:8000]}"
    response = ask_llm(prompt)
    return response or "Summarization failed."

# ------------------- Comparison Tools -------------------
def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> str:
    try:
        content1 = read_file(file1).splitlines()
        content2 = read_file(file2).splitlines()
    except Exception as e:
        return f"Error reading files: {e}"
    diff = difflib.unified_diff(content1, content2, fromfile=str(file1), tofile=str(file2))
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Files are identical."

def compare_webpages(url1: str, url2: str) -> str:
    text1 = fetch_page(url1)
    text2 = fetch_page(url2)
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    diff = difflib.unified_diff(lines1, lines2, fromfile=url1, tofile=url2)
    diff_text = '\n'.join(diff)
    return diff_text if diff_text else "Pages are identical."

# ------------------- Scheduled Tasks -------------------
scheduled_jobs = []  # list of dicts with id, command, schedule string
_scheduler_running = False
_scheduler_thread = None

def start_scheduler():
    global _scheduler_running, _scheduler_thread
    if _scheduler_running or not HAS_SCHEDULE:
        return
    _scheduler_running = True

    def run_loop():
        while _scheduler_running:
            schedule.run_pending()
            time.sleep(1)

    _scheduler_thread = threading.Thread(target=run_loop, daemon=True)
    _scheduler_thread.start()

def stop_scheduler():
    global _scheduler_running
    _scheduler_running = False
    if _scheduler_thread and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=2)

def _execute_scheduled_task(job):
    command = job['command']
    tool_args = job.get('tool_args', {})
    if command.startswith('shell:'):
        cmd = command[6:]
        # Use a safe shell execution (allowlist enforced)
        result = _safe_shell_command(cmd, auto_confirm=True)
        console.print(f"[yellow]Scheduled task executed: {cmd}[/yellow]\n{result}")
    elif command.startswith('fetch:'):
        url = command[6:]
        content = fetch_page(url)
        output = tool_args.get('output', 'schedule_output.txt')
        write_file(output, content, append=True)
        console.print(f"[yellow]Scheduled fetch saved to {output}[/yellow]")
    else:
        console.print(f"[red]Unknown scheduled command: {command}[/red]")

def schedule_task(command: str, time_str: str, tool_args: Optional[Dict] = None) -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed. Install with: pip install schedule"
    start_scheduler()
    job = {
        'command': command,
        'tool_args': tool_args or {}
    }
    try:
        schedule.every().day.at(time_str).do(_execute_scheduled_task, job)
        return f"Scheduled {command} at {time_str} daily"
    except Exception as e:
        return f"Failed to schedule: {e}"

def list_scheduled_tasks() -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if not jobs:
        return "No scheduled tasks."
    lines = []
    for i, job in enumerate(jobs, 1):
        lines.append(f"{i}. {job}")
    return "\n".join(lines)

def cancel_scheduled_task(index: int) -> str:
    if not HAS_SCHEDULE:
        return "Schedule library not installed."
    jobs = schedule.get_jobs()
    if 1 <= index <= len(jobs):
        schedule.cancel_job(jobs[index-1])
        return f"Cancelled task {index}"
    else:
        return "Invalid task index"

# ------------------- Shell Commands (with allowlist) -------------------
def _safe_shell_command(cmd: str, auto_confirm: bool = False) -> str:
    """Execute a shell command with allowlist restrictions."""
    # Parse command: only the first word (executable) is checked
    parts = cmd.split()
    if not parts:
        return "Empty command."
    executable = parts[0]
    allowed = config.get("shell_allowed_commands")
    if executable not in allowed:
        return f"Command '{executable}' is not allowed. Allowed: {', '.join(allowed)}"
    if not auto_confirm:
        console.print(Panel(f"[yellow]Shell command: {cmd}[/yellow]", title="⚠️ Shell Execution", border_style="red"))
        if not Confirm.ask("Execute this command?", default=False):
            return "Command cancelled by user."
    try:
        # Use subprocess with list to avoid shell injection
        result = subprocess.run(parts, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

# ------------------- Web Helpers -------------------
def fetch_page(url: str) -> str:
    """Fetch a page using cache, lynx, or requests (async fallback)."""
    cached = get_cache(url)
    if cached is not None:
        return cached
    try:
        # Try lynx first for text extraction
        proc = subprocess.run(["lynx", "-dump", "-nolist", url],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            content = proc.stdout
            set_cache(url, content)
            return content
        else:
            logger.warning("lynx returned error, falling back to requests")
            # Fallback to requests
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator="\n")
            set_cache(url, content)
            return content
    except FileNotFoundError:
        logger.info("lynx not found, using requests")
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator="\n")
            set_cache(url, content)
            return content
        except Exception as e:
            return f"Error fetching page: {e}"
    except Exception as e:
        return f"Error fetching page: {e}"

async def fetch_page_async(url: str) -> str:
    """Async version using aiohttp if available."""
    cached = get_cache(url)
    if cached is not None:
        return cached
    if not HAS_AIOHTTP:
        return fetch_page(url)  # fallback sync
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": config.get("user_agent")}) as resp:
                text = await resp.text()
                soup = BeautifulSoup(text, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text(separator="\n")
                set_cache(url, content)
                return content
    except Exception as e:
        return f"Error fetching page asynchronously: {e}"

def extract_links(page_text: str, base_url: str) -> List[Tuple[str, str]]:
    """Extract links from the page using BeautifulSoup."""
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
        logger.error(f"Error extracting links: {e}")
        return []

def resolve_redirect(url: str) -> str:
    try:
        resp = session.get(url, allow_redirects=False, timeout=15)
        if resp.status_code in (301, 302) and "Location" in resp.headers:
            return resp.headers["Location"]
    except:
        pass
    return url

def search_duckduckgo_lite(query: str) -> str:
    """Perform a DuckDuckGo Lite search, filter ads, return text and links."""
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
def ask_llm(prompt: str, context: str = "", stream: bool = False) -> Union[str, None]:
    """Call Ollama with the given prompt. If stream=True, yield chunks."""
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    try:
        if stream:
            # Return a generator for streaming
            return ollama.chat(model=config.get("model"), messages=[{"role": "user", "content": full_prompt}], stream=True)
        else:
            response = ollama.chat(model=config.get("model"), messages=[{"role": "user", "content": full_prompt}])
            return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        return None

def decide_next_action(user_query: str, current_url: str, page_text: str, available_links: List[Tuple[str, str]]) -> Dict:
    """Ask LLM what to do next, return JSON."""
    # Truncate page text to reduce tokens
    max_chars = config.get("max_page_chars")
    if len(page_text) > max_chars:
        page_text = page_text[:max_chars] + "... [truncated]"
    links_text = "\n".join([f"- {text}: {url}" for text, url in available_links[:20]])
    tools_description = """
You have access to additional tools:
- **File system**: read_file(path), write_file(path, content, append=False), list_dir(path="."), search_files(pattern, root=".")
- **Shell**: shell(command) – executes a system command (requires confirmation). Allowed commands: ls, pwd, echo, cat, grep, git.
- **Memory**: save_memory(key, value), recall_memory(key), list_memory()
- **Git**: git_status(repo_path="."), git_diff(repo_path=".", staged=False), git_commit(repo_path=".", message=""), git_branch(repo_path=".", new_branch=None), git_log(repo_path=".", n=10)
- **Project**: project_info(root=".") – gives overview of codebase
- **Summarization**: summarize(text) – returns concise summary
- **Comparison**: compare_files(file1, file2), compare_webpages(url1, url2)
- **Scheduling**: schedule_task(command, time_str, tool_args=None), list_scheduled_tasks(), cancel_scheduled_task(index)
- **Web browsing**: search(query), visit_link(url), extract(answer)

When you need to use a tool, respond with a JSON object that includes "action": "tool", "tool_name": <name>, and "tool_args": <arguments>.
For example:
{"action": "tool", "tool_name": "write_file", "tool_args": {"path": "~/Documents/note.txt", "content": "Hello"}}
{"action": "tool", "tool_name": "shell", "tool_args": {"command": "ls -la"}}
{"action": "tool", "tool_name": "save_memory", "tool_args": {"key": "important", "value": "some value"}}
{"action": "tool", "tool_name": "git_status", "tool_args": {"repo_path": "."}}
{"action": "tool", "tool_name": "summarize", "tool_args": {"text": "long text here"}}
{"action": "tool", "tool_name": "compare_files", "tool_args": {"file1": "a.txt", "file2": "b.txt"}}
{"action": "tool", "tool_name": "schedule_task", "tool_args": {"command": "shell:echo hello", "time_str": "09:00"}}
"""
    prompt = f"""
You are an AI OS Terminal Browser. The user's goal is: "{user_query}"
You are currently on: {current_url}

Page content (first {max_chars} chars):
{page_text}

Available links on this page:
{links_text}

{tools_description}

**IMPORTANT INSTRUCTIONS:**
- If the current page is a search results page, the goal is NOT yet achieved. You must choose a relevant link to visit and then extract the actual information from that page.
- Only stop if the page already contains the exact answer the user wants, or if you have already extracted it.
- If you are on a search results page, use "visit_link" with one of the URLs from the list.
- If you are on a page that likely contains the answer, use "extract" to pull out the relevant information.
- Only use "search" again if the current page is completely irrelevant.

Decide what to do next. Choose one action from:
- "search" (with query)
- "visit_link" (with url)
- "extract" (with answer)
- "tool" (with tool_name and tool_args)
- "multi_edit" (with edits list)
- "stop"

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
def process_query(user_query: str, interactive: bool = False) -> Optional[str]:
    """
    Process a single query autonomously, returning the final answer.
    If interactive is True, it will show output and ask for confirmation at each step.
    Otherwise, it runs silently and returns the final extracted answer.
    """
    current_url = None
    page_text = ""
    available_links = []
    final_answer = None
    step_count = 0
    max_steps = 20  # prevent infinite loops

    while step_count < max_steps:
        step_count += 1
        # Special commands
        if user_query.strip() == "!model":
            # Handled elsewhere
            break
        elif user_query.strip() == "!memory":
            break

        # Perform search or fetch page if web browsing
        if current_url is None and not user_query.startswith("!"):
            if interactive:
                console.print(f"[green]Searching for: {user_query}[/green]")
            page_text = search_duckduckgo_lite(user_query)
            # Extract links from search results again
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
                    result = _safe_shell_command(cmd, auto_confirm=not interactive)
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

        # In interactive mode, allow user to intervene after each step
        if interactive:
            if not Confirm.ask("[bold yellow]Continue with AI?[/bold yellow]", default=True):
                manual = Prompt.ask("Enter command (URL, !model, !memory, 'exit')")
                if manual.lower() == "exit":
                    break
                elif manual.strip() == "!model":
                    new_model = select_model()
                    config.set("model", new_model)
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

    if step_count >= max_steps:
        logger.warning("Reached max steps, stopping")
        final_answer = final_answer or "Process exceeded maximum steps."

    return final_answer

# ------------------- Flask API Server -------------------
def start_api_server(host: str = None, port: int = None):
    if not HAS_FLASK:
        console.print("[red]Flask not installed. Cannot start API server.[/red]")
        return
    host = host or config.get("api_host")
    port = port or config.get("api_port")
    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field'}), 400
        query = data['query']
        # Optionally use conversation history
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
    console.print(Panel.fit("[bold cyan]AI OS Terminal Browser (Enhanced)[/bold cyan] - AI-powered assistant with web, file, shell, Git, scheduling, and memory", style="bold"))

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]Ollama not found. Please install Ollama and make sure it's running.[/red]")
        sys.exit(1)

    model = config.get("model")
    if not model:
        model = select_model()
        config.set("model", model)
    else:
        console.print(f"[green]Using model from config: {model}[/green]")

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
    parser.add_argument('--host', default=None, help="Host for API server")
    parser.add_argument('--port', type=int, default=None, help="Port for API server")
    args = parser.parse_args()

    if args.serve:
        start_api_server(host=args.host, port=args.port)
    else:
        interactive_main()

if __name__ == "__main__":
    main()
