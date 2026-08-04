# Kubernetes Agent — cluster dashboard

Docker-Desktop-style dashboard for a conversational agent that manages a whole Kubernetes namespace, not just training. Three levels: pick a **cluster** (kubeconfig context — one per Minikube profile or any other cluster you've configured) → pick a **namespace** within it ("project") → chat scoped to just that namespace. Destructive actions (delete, scale, restart, launching a new training job) always pause and ask for your explicit go-ahead before running.

This is a scope expansion of [ml-agent-mvp-chat](../ml-agent-mvp-chat) (training-only, single hardcoded namespace, CLI). Same underlying patterns — function calling, per-turn conversation memory, lazy Kubernetes client loading — extended to more resource types, per-namespace isolation, multi-cluster support, and a browser UI.

## Files

- `Dockerfile` + `entrypoint.sh` — Universal Training Image (unchanged from the original project; clones a repo at runtime, runs `train.py`)
- `agent_core.py` — all agent logic: namespace-scoped tool factory, tool-schema generation from docstrings, confirmation-flow state machine (`run_turn` / `resume_turn`), no UI code
- `app.py` — Streamlit dashboard (cluster list → namespace list → per-namespace chat)
- `requirements.txt` — dependencies

## LLM backend: Groq

Uses Groq's OpenAI-compatible `chat.completions` API — a genuinely free tier (not a trial), fast inference, solid function-calling support. Get a free key at [console.groq.com/keys](https://console.groq.com/keys). The model is a plain text field in the sidebar (default `llama-3.3-70b-versatile`) rather than hardcoded, because providers keep retiring/renaming model IDs — if you hit a 404, check [console.groq.com/docs/models](https://console.groq.com/docs/models) and paste in a different one, no code change needed.

Tool schemas (`agent_core._build_tool_schema`) are generated automatically from each tool function's type hints and Google-style docstring — adding a new tool to `build_tools()` doesn't require hand-writing a JSON schema for it.

## How isolation works

Tools are built per-namespace *and* per-cluster by `agent_core.build_tools(namespace, context)` — both are Python closure variables, **not** arguments the model can set. Opening a chat for namespace `training` in cluster `minikube` gives the LLM tools that can only ever touch that exact namespace in that exact cluster; it has no way to ask for a different one mid-conversation.

## How confirmation works

Tools are split into read-only (execute immediately) and destructive (`agent_core.CONFIRM_TOOLS`, 16 of the 46). When the model wants to run a destructive tool, `run_turn` stops and returns a `confirm` result instead of executing it; the UI shows exactly what's about to happen and waits for **Zatwierdź** / **Odrzuć** before `resume_turn` actually runs it (or tells the model you declined).

## What's covered (46 tools + namespace management)

**Namespaces:** create/delete are direct dashboard actions (no chat, no LLM) — see below.

**Diagnostics (read-only):** `list_pods`, `describe_pod` (equivalent to `kubectl describe pod` — surfaces CrashLoopBackOff/OOMKilled/ImagePullBackOff, restart counts, and *why a Pending pod hasn't been scheduled*, e.g. insufficient GPU), `get_pod_logs`, `get_recent_events`, `get_resource_usage` (live CPU/memory via metrics-server), `check_service_endpoints` (detects the classic Service-selector/pod-label mismatch), `check_gpu_availability`, `list_nodes`, `list_resource_quotas`, `list_limit_ranges`, `list_pod_disruption_budgets`, `list_crds`.

**Workloads:** `list_deployments`/`scale_deployment`/`restart_deployment`/`create_deployment` (general-purpose, no GPU - run anything, e.g. `nginx`), `list_statefulsets`/`scale_statefulset`/`restart_statefulset`, `list_daemonsets`/`restart_daemonset`, `list_jobs`/`create_training_job`/`delete_job`/`get_job_status`/`get_job_logs` (GPU-specific, ML training from a GitHub repo), `delete_pod`.

**Networking:** `list_services`, `list_ingresses`/`delete_ingress`, `list_network_policies`.

**Config & storage:** `list_configmaps`/`get_configmap`/`delete_configmap`, `list_secrets`/`delete_secret` (names/types only — see below), `list_pvcs`/`delete_pvc`, `list_storage_classes`, `list_hpas`.

**RBAC (read-only):** `list_service_accounts`, `list_roles`, `list_role_bindings`.

**Cluster:** `list_nodes`, `cordon_node`, `uncordon_node`, `drain_node` (cordon + evict all evictable pods, respecting PodDisruptionBudgets, skipping DaemonSet-managed pods) — the one deliberate exception to namespace isolation, since Nodes aren't namespaced objects in Kubernetes at all.

## Namespace management

Creating/deleting a namespace is a **dashboard action, not a chat tool** (`agent_core.create_namespace`/`delete_namespace`, called directly from `_namespace_view` in `app.py`). Deliberately kept out of the LLM's hands: deleting a namespace destroys everything inside it, and unlike every other destructive tool here it isn't scoped to the namespace you're chatting in — there's no "the LLM can't reach outside its box" guarantee to fall back on. Delete requires its own explicit two-step UI confirmation before it runs.

## Deliberately NOT included

- **`kubectl exec` (arbitrary command execution inside a container).** Every other destructive tool here has a bounded, auditable effect (you can read exactly what `delete_pod` or `scale_deployment` will do before approving it). Arbitrary exec is a different risk category — the command itself could do anything, including reading mounted secrets or exfiltrating data — so it's excluded rather than gated behind the same confirmation UI, which wouldn't give you a real way to audit it beforehand.
- **Secret values.** `list_secrets` shows name and type only; nothing ever reads or writes actual secret data through the LLM, since that data would be sent to Groq's API as part of the conversation.
- **RBAC writes** (creating/modifying Roles, RoleBindings, ServiceAccounts) — misconfiguring these can lock you out or open a privilege-escalation hole; read-only only.
- **Deployment rollback/rollout history.** Correctly reimplementing `kubectl rollout undo` means walking ReplicaSet revision history, not a single API call — not yet built, tracked as a known gap rather than shipped half-right.
- **Cross-cluster config-drift detection, cost optimization, Helm, CI/CD/GitOps.** Different tool categories entirely (need historical data, billing APIs, or a `helm` binary/subprocess) — out of scope for a per-namespace chat agent.

## Deployment (WSL2 Ubuntu)

```bash
# 1. Start Minikube with GPU support
minikube start --driver docker --gpus all

# 2. Build the Universal Image INSIDE Minikube's Docker daemon
#    (required because agent_core.py uses imagePullPolicy: Never)
eval $(minikube docker-env)
docker build -t ml-trainer:latest .

# 3. Install dependencies (host / venv)
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), paste a free Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) into the sidebar, pick a cluster, then a namespace, and start chatting.

**Do not expose this on a public IP.** The Streamlit process has whatever Kubernetes permissions your `~/.kube/config` grants — anyone who can reach the app (with their own, unrelated API key) can approve destructive actions against your real cluster. Run it bound to localhost only.

## Notes

- Cluster and namespace list views are plain cluster reads — they work without an API key; you only need one to open a namespace's chat.
- Going back a level and returning keeps that namespace's conversation (and any pending confirmation) exactly where you left it.
- `get_job_logs` / `get_pod_logs` clamp `tail_lines` to `agent_core.MAX_LOG_TAIL_LINES` (500) so a runaway request can't dump a huge log into the model's context.
- `MAX_TOOL_ITERATIONS` (8) caps how many tool calls the model can chain in a single turn. `_generate_with_retry` retries transient errors (rate limits, server overload) with exponential backoff, but fails immediately on a bad model name or bad key.

## Keeping token cost down

These APIs are stateless - every single request resends the *entire* conversation, so an unbounded chat means unbounded, ever-growing cost per message (this is what burned a whole day's free quota in one long troubleshooting session). Levers, roughly biggest-to-smallest:

- **`agent_core.MAX_HISTORY_MESSAGES` (40)** — hard cap on how much history actually gets sent to the API per call, applied in `_generate_with_retry`/`_trim_history`. The full conversation still stays in the UI for scrollback; only what's *sent* is capped, cut cleanly at a user-message boundary so a tool call and its result never get split apart.
- **Pick a smaller/cheaper model in the sidebar** (e.g. `llama-3.1-8b-instant` instead of `llama-3.3-70b-versatile`) — free-tier daily token quotas are per-model, so switching also sidesteps a quota you've already exhausted. Trade-off: smaller models are more prone to the kind of tool-use mistakes described above (fabricated arguments, pointless tool calls) - cheaper isn't free if it just means more back-and-forth to get the same result.
- **`MAX_LOG_TAIL_LINES` (500) / `MAX_EVENTS` (20)** already clamp the biggest individual tool results (see above) - lower them further if logs/events are consistently the biggest single contributor to a conversation's size.
- Tool descriptions carry real behavioral guidance now (e.g. the node-op warnings, the anti-hallucination rules) - deliberately not trimmed for size, since that guidance is what fixed actual bugs earlier. Shortening them would trade cost for correctness; do that consciously, not as a blanket optimization.
- The training repo must have `train.py` in its root; `requirements.txt` is optional. `backoff_limit=0`: failed jobs are not retried (fail fast, MVP behavior). Private repos: use a URL with a token (`https://<TOKEN>@github.com/user/repo`).
