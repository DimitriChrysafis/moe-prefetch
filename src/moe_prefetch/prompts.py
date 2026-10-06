from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

EVAL_FRACTION = 0.25


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = field(default=())

    def render(self, tokenizer) -> str:
        return tokenizer.apply_chat_template(
            list(self.messages),
            tools=list(self.tools) or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(self, tokenizer, max_tokens: int | None = None) -> list[int]:
        ids = list(
            tokenizer.apply_chat_template(
                list(self.messages),
                tools=list(self.tools) or None,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if max_tokens is not None and len(ids) > max_tokens:
            ids = ids[-max_tokens:]
        return ids


def _user(text: str) -> tuple[dict[str, Any], ...]:
    return ({"role": "user", "content": text},)


def _tool(name: str, description: str, **params: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": value} for key, value in params.items()
                },
                "required": list(params),
            },
        },
    }


def _call(name: str, **arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _result(content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content, indent=1)
    return {"role": "tool", "content": text}


CODE_TOOLS = (
    _tool("read_file", "Read a file from the repository", path="Path relative to repo root"),
    _tool(
        "write_file",
        "Overwrite a file with new contents",
        path="Path relative to repo root",
        content="Full file contents",
    ),
    _tool(
        "run_shell", "Run a shell command in the repo and return stdout and stderr", cmd="Command"
    ),
    _tool("search", "Search the repository with a regular expression", pattern="Regex pattern"),
)

WEB_TOOLS = (
    _tool("web_search", "Search the web and return the top results", query="Search query"),
    _tool("fetch_url", "Fetch a page and return its text", url="Absolute URL"),
    _tool("save_note", "Save a note for the user", title="Short title", body="Markdown body"),
)

DATA_TOOLS = (
    _tool("sql", "Run a read only SQL query against the analytics warehouse", query="SQL"),
    _tool(
        "plot",
        "Render a chart from a query result",
        query="SQL returning the data",
        kind="line, bar or scatter",
    ),
    _tool("send_email", "Draft an email for the user to review", to="Recipient", body="Body"),
)

OPS_TOOLS = (
    _tool("kubectl", "Run a kubectl command against the staging cluster", args="Arguments"),
    _tool("get_logs", "Fetch recent logs for a service", service="Service name", since="Window"),
    _tool("get_metrics", "Query Prometheus", promql="PromQL expression", range="Time range"),
    _tool("page_oncall", "Page the on call engineer", summary="One line summary"),
)

CALENDAR_TOOLS = (
    _tool("list_events", "List calendar events in a date range", start="ISO date", end="ISO date"),
    _tool(
        "create_event",
        "Create a calendar event",
        title="Event title",
        start="ISO datetime",
        end="ISO datetime",
    ),
    _tool(
        "find_free_slots", "Find free slots for a set of people", people="Comma separated emails"
    ),
)

SYSTEM_CODE = (
    "You are a careful software engineering agent working inside a git repository. "
    "Prefer small, verifiable changes. Read before you write, run the tests after every "
    "change, and explain what you did in two or three sentences when you are finished."
)

PROMPTS: tuple[Prompt, ...] = (
    Prompt(
        "code-py-lru",
        "code",
        _user(
            "Write a Python class LRUCache with get and put in O(1) using only the standard "
            "library, plus a few pytest tests covering eviction order and updates."
        ),
    ),
    Prompt(
        "code-py-explain-gen",
        "code",
        _user(
            "Explain what this Python does and when it would be a bad idea:\n\n"
            "def chunks(it, n):\n"
            "    it = iter(it)\n"
            "    while batch := list(itertools.islice(it, n)):\n"
            "        yield batch\n\n"
            "def merge(*sorted_iters):\n"
            "    return heapq.merge(*sorted_iters, key=lambda r: (r.ts, r.id))\n"
        ),
    ),
    Prompt(
        "code-py-debug-async",
        "code",
        _user(
            "This script is supposed to fetch 100 URLs with at most 10 in flight, but it "
            "runs them one at a time. Find the bug and fix it.\n\n"
            "import asyncio, aiohttp\n\n"
            "async def fetch(session, sem, url):\n"
            "    async with sem:\n"
            "        async with session.get(url) as r:\n"
            "            return await r.text()\n\n"
            "async def main(urls):\n"
            "    sem = asyncio.Semaphore(10)\n"
            "    async with aiohttp.ClientSession() as s:\n"
            "        out = []\n"
            "        for u in urls:\n"
            "            out.append(await fetch(s, sem, u))\n"
            "        return out\n"
        ),
    ),
    Prompt(
        "code-py-pandas",
        "code",
        _user(
            "I have a pandas DataFrame of trades with columns ts, symbol, price, qty. Write a "
            "function that returns per symbol 5 minute VWAP bars, with empty bars forward "
            "filled, and explain how you handle timezone aware timestamps."
        ),
    ),
    Prompt(
        "code-c-ringbuf",
        "code",
        _user(
            "Write a lock free single producer single consumer ring buffer in C11 using "
            "stdatomic.h. Use a power of two capacity, and explain which memory orders you "
            "chose for head and tail and why."
        ),
    ),
    Prompt(
        "code-c-debug-strtok",
        "code",
        _user(
            "Why does this C program sometimes crash, and how should it be written?\n\n"
            "char *first_word(const char *s) {\n"
            "    char buf[64];\n"
            "    strcpy(buf, s);\n"
            '    return strtok(buf, " ");\n'
            "}\n\n"
            "int main(void) {\n"
            '    char *w = first_word("hello world from c");\n'
            '    printf("%s\\n", w);\n'
            "}\n"
        ),
    ),
    Prompt(
        "code-ts-debounce",
        "code",
        _user(
            "Implement a typed debounce function in TypeScript that preserves the argument "
            "types of the wrapped function, supports leading and trailing edges, and exposes "
            "cancel and flush methods. Include a short usage example in a React component."
        ),
    ),
    Prompt(
        "code-ts-explain-types",
        "code",
        _user(
            "Explain this TypeScript type step by step:\n\n"
            "type DeepPartial<T> = T extends Function\n"
            "  ? T\n"
            "  : T extends Array<infer U>\n"
            "  ? Array<DeepPartial<U>>\n"
            "  : T extends object\n"
            "  ? { [K in keyof T]?: DeepPartial<T[K]> }\n"
            "  : T;\n"
        ),
    ),
    Prompt(
        "code-rust-vs-go",
        "code",
        _user(
            "Rewrite this Go worker pool in Rust using tokio, keeping the same semantics:\n\n"
            "func pool(jobs <-chan int, n int) <-chan int {\n"
            "    out := make(chan int)\n"
            "    var wg sync.WaitGroup\n"
            "    for i := 0; i < n; i++ {\n"
            "        wg.Add(1)\n"
            "        go func() { defer wg.Done(); for j := range jobs { out <- j * j } }()\n"
            "    }\n"
            "    go func() { wg.Wait(); close(out) }()\n"
            "    return out\n"
            "}\n"
        ),
    ),
    Prompt(
        "code-sql-window",
        "code",
        _user(
            "Given tables users(id, signup_at) and events(user_id, ts, kind), write a "
            "Postgres query that computes weekly retention cohorts for the last 12 weeks, "
            "and explain how to make it fast on 500 million events."
        ),
    ),
    Prompt(
        "prose-ssd",
        "prose",
        _user(
            "Explain to a new backend engineer why random 4 KB reads on an NVMe SSD are so "
            "much slower than sequential reads, and what that means for designing a cache."
        ),
    ),
    Prompt(
        "prose-history-printing",
        "prose",
        _user(
            "Write a short essay on how the printing press changed the way scientific "
            "knowledge spread in Europe between 1450 and 1700."
        ),
    ),
    Prompt(
        "prose-ironman",
        "prose",
        _user(
            "I am training for my first full Ironman in 30 weeks and currently run 25 miles "
            "a week. Outline a periodized plan for swim, bike and run, and explain how to "
            "avoid overtraining."
        ),
    ),
    Prompt(
        "prose-transformers",
        "prose",
        _user(
            "Explain how attention works in a transformer to someone who knows linear "
            "algebra but has never studied machine learning. Use one concrete example."
        ),
    ),
    Prompt(
        "prose-email",
        "prose",
        _user(
            "Draft a polite but firm email to a landlord asking them to fix a broken heater "
            "that has been reported three times over two weeks."
        ),
    ),
    Prompt(
        "prose-story",
        "prose",
        _user(
            "Write a 300 word short story about a lighthouse keeper who discovers the light "
            "has been signaling to something other than ships."
        ),
    ),
    Prompt(
        "prose-compare-db",
        "prose",
        _user(
            "Compare Postgres, SQLite and DuckDB for a small analytics product. Cover "
            "concurrency, deployment, query speed and when each one is the wrong choice."
        ),
    ),
    Prompt(
        "math-series",
        "math",
        _user(
            "Prove that the sum of 1/n^2 for n from 1 to infinity converges, "
            "and find a tight upper bound."
        ),
    ),
    Prompt(
        "math-probability",
        "math",
        _user(
            "A drawer has 6 red and 4 blue socks. You draw socks one at a time without "
            "replacement until you have a matching pair. What is the expected number of draws?"
        ),
    ),
    Prompt(
        "math-linalg",
        "math",
        _user(
            "Let A be a real symmetric 3x3 matrix with eigenvalues 1, 2 and 4. Compute the "
            "determinant and trace of A^2 - 3A + 2I, and explain each step."
        ),
    ),
    Prompt(
        "math-calculus",
        "math",
        _user("Evaluate the integral of x^2 * e^(-x) from 0 to infinity in two different ways."),
    ),
    Prompt(
        "math-number-theory",
        "math",
        _user(
            "Find all integer solutions to x^2 - 7y^2 = 1 with 0 < x < 1000 and explain the method."
        ),
    ),
    Prompt(
        "math-word",
        "math",
        _user(
            "Two trains leave stations 300 km apart at the same time heading toward each "
            "other at 70 km/h and 80 km/h. A bird flies back and forth between them at "
            "120 km/h until they meet. How far does the bird fly? Then solve it the hard way."
        ),
    ),
    Prompt(
        "agent-fix-test",
        "agent",
        (
            {"role": "system", "content": SYSTEM_CODE},
            {
                "role": "user",
                "content": "The test tests/test_parse.py::test_dates fails on CI. Fix it.",
            },
            _call("run_shell", cmd="pytest tests/test_parse.py::test_dates -x -q"),
            _result(
                "F\n=== FAILURES ===\n___ test_dates ___\n"
                "    def test_dates():\n"
                ">       assert parse_date('2024-03-10T02:30') == datetime(2024, 3, 10, 2, 30)\n"
                "E       ValueError: time data '2024-03-10T02:30' does not match format "
                "'%Y-%m-%d %H:%M'\n"
                "src/app/parse.py:14: ValueError\n1 failed in 0.21s"
            ),
            _call("read_file", path="src/app/parse.py"),
            _result(
                "from datetime import datetime\n\n"
                "FORMATS = ['%Y-%m-%d %H:%M', '%Y-%m-%d']\n\n"
                "def parse_date(s):\n"
                "    for fmt in FORMATS[:-1]:\n"
                "        return datetime.strptime(s, fmt)\n"
                "    return datetime.strptime(s, FORMATS[-1])\n"
            ),
        ),
        CODE_TOOLS,
    ),
    Prompt(
        "agent-refactor",
        "agent",
        (
            {"role": "system", "content": SYSTEM_CODE},
            {
                "role": "user",
                "content": "Replace every use of the deprecated requests_retry helper with "
                "httpx and a tenacity retry decorator. Keep behavior the same.",
            },
            _call("search", pattern="requests_retry"),
            _result(
                "src/clients/billing.py:3:from util.http import requests_retry\n"
                "src/clients/billing.py:21:    r = requests_retry(retries=5).get(url, timeout=10)\n"
                "src/clients/geo.py:2:from util.http import requests_retry\n"
                "src/clients/geo.py:40:    resp = requests_retry().post(GEO_URL, json=payload)\n"
                "src/util/http.py:8:def requests_retry(retries=3, backoff=0.3):"
            ),
        ),
        CODE_TOOLS,
    ),
    Prompt(
        "agent-research",
        "agent",
        (
            {
                "role": "system",
                "content": "You are a research assistant. Cite sources with URLs and save a "
                "concise note when the user's question is answered.",
            },
            {
                "role": "user",
                "content": "What are the main approaches to speculative decoding for LLMs, "
                "and which ones work without a separate draft model?",
            },
            _call("web_search", query="speculative decoding without draft model"),
            _result(
                [
                    {
                        "title": "Medusa: Simple LLM inference acceleration with multiple "
                        "decoding heads",
                        "url": "https://arxiv.org/abs/2401.10774",
                        "snippet": "Adds extra decoding heads to predict multiple subsequent "
                        "tokens in parallel, verified with tree attention.",
                    },
                    {
                        "title": "Lookahead decoding",
                        "url": "https://arxiv.org/abs/2402.02057",
                        "snippet": "Uses Jacobi iteration to generate n-grams in parallel and "
                        "verifies them, no draft model required.",
                    },
                    {
                        "title": "EAGLE: speculative sampling requires rethinking feature "
                        "uncertainty",
                        "url": "https://arxiv.org/abs/2401.15077",
                        "snippet": "Autoregression at the feature level with a lightweight head.",
                    },
                ]
            ),
        ),
        WEB_TOOLS,
    ),
    Prompt(
        "agent-analytics",
        "agent",
        (
            {
                "role": "system",
                "content": "You are a data analyst with read only warehouse access. Tables: "
                "orders(id, user_id, created_at, total_cents, status), users(id, country, "
                "created_at), refunds(order_id, amount_cents, created_at).",
            },
            {
                "role": "user",
                "content": "Revenue dropped last week. Figure out whether it was fewer "
                "orders, smaller orders, or more refunds, and break it down by country.",
            },
            _call(
                "sql",
                query="select date_trunc('week', created_at) wk, count(*), sum(total_cents) "
                "from orders where created_at > now() - interval '3 weeks' group by 1 order by 1",
            ),
            _result(
                [
                    {"wk": "2025-09-15", "count": 18234, "sum": 104882113},
                    {"wk": "2025-09-22", "count": 18410, "sum": 106001942},
                    {"wk": "2025-09-29", "count": 15102, "sum": 88123550},
                ]
            ),
        ),
        DATA_TOOLS,
    ),
    Prompt(
        "agent-incident",
        "agent",
        (
            {
                "role": "system",
                "content": "You are an SRE assistant for a staging cluster. Investigate before "
                "acting. Only page on call for customer facing impact.",
            },
            {
                "role": "user",
                "content": "checkout-api p99 latency alert just fired. What is going on?",
            },
            _call(
                "get_metrics",
                promql="histogram_quantile(0.99, sum by (le) (rate(http_request_seconds_bucket"
                '{service="checkout-api"}[5m])))',
                range="1h",
            ),
            _result({"series": [0.21, 0.22, 0.24, 0.8, 2.9, 3.4, 3.1], "step": "10m"}),
            _call("get_logs", service="checkout-api", since="30m"),
            _result(
                "WARN pool exhausted waiting for connection (pg-primary) waited=2.8s\n"
                "WARN pool exhausted waiting for connection (pg-primary) waited=3.1s\n"
                "INFO deploy checkout-api@4f2a1c rolled out 3/3 pods\n"
                "ERROR context deadline exceeded calling inventory-svc"
            ),
        ),
        OPS_TOOLS,
    ),
    Prompt(
        "agent-calendar",
        "agent",
        (
            {
                "role": "system",
                "content": "You manage the user's calendar. The user is in America/Chicago. "
                "Never double book and keep 15 minutes between meetings.",
            },
            {
                "role": "user",
                "content": "Set up a 45 minute design review with ana@acme.io and "
                "raj@acme.io sometime Thursday or Friday afternoon.",
            },
            _call("find_free_slots", people="me,ana@acme.io,raj@acme.io"),
            _result(
                {
                    "thursday": [["13:00", "13:30"], ["15:15", "17:00"]],
                    "friday": [["14:00", "16:00"]],
                    "timezone": "America/Chicago",
                }
            ),
        ),
        CALENDAR_TOOLS,
    ),
    Prompt(
        "agent-ts-build",
        "agent",
        (
            {"role": "system", "content": SYSTEM_CODE},
            {
                "role": "user",
                "content": "npm run build is broken after the dependency bump. Get it green.",
            },
            _call("run_shell", cmd="npm run build"),
            _result(
                "> web@2.4.0 build\n> tsc -p . && vite build\n\n"
                "src/api/client.ts:42:17 - error TS2345: Argument of type 'string | undefined' "
                "is not assignable to parameter of type 'string'.\n"
                "src/hooks/useUser.ts:9:10 - error TS2305: Module '\"@tanstack/react-query\"' "
                "has no exported member 'useQuery'.\n\nFound 2 errors in 2 files."
            ),
        ),
        CODE_TOOLS,
    ),
)


def _rank(prompt_id: str) -> str:
    return hashlib.sha256(prompt_id.encode()).hexdigest()


def _eval_ids() -> frozenset[str]:
    out = set()
    for category in sorted({p.category for p in PROMPTS}):
        ids = sorted((p.id for p in PROMPTS if p.category == category), key=_rank)
        out.update(ids[: max(1, round(len(ids) * EVAL_FRACTION))])
    return frozenset(out)


EVAL_IDS = _eval_ids()


def split_of(prompt_id: str) -> str:
    return "eval" if prompt_id in EVAL_IDS else "train"


def get_prompts(split: str = "all") -> list[Prompt]:
    if split not in {"all", "train", "eval"}:
        raise ValueError(f"unknown split {split!r}, expected all, train or eval")
    return [p for p in PROMPTS if split == "all" or split_of(p.id) == split]


def get_prompt(prompt_id: str) -> Prompt:
    for prompt in PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise KeyError(f"no prompt with id {prompt_id!r}")
