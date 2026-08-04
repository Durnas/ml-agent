"""
app.py - Docker-Desktop-style dashboard for the cluster-management agent.

Three levels: pick a cluster (kubeconfig context - one per Minikube profile
or any other cluster you have configured) -> pick a namespace within it
("project") -> chat scoped to just that namespace. agent_core.build_tools
enforces the namespace isolation - the LLM never sees a namespace argument
to pick from. Going back a level and returning keeps everything (chat
history, pending confirmations) exactly where you left it.

Destructive tool calls (agent_core.CONFIRM_TOOLS) pause for an explicit
Zatwierdź/Odrzuć before running - see agent_core.run_turn / resume_turn.

Run: streamlit run app.py
"""
import json
import sys

import groq
import streamlit as st

import agent_core

# On some WSL2/Linux setups LANG isn't set to a UTF-8 locale, which makes
# Python fall back to ASCII for stdout/stderr - any library that logs a
# Polish message (or any non-ASCII text) then raises UnicodeEncodeError.
# Reconfiguring here fixes it regardless of the shell's locale.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # stream doesn't support reconfigure (e.g. already redirected) - not fatal

st.set_page_config(page_title="Kubernetes Agent", page_icon="☸️", layout="wide")

_STATUS_COLOR = {
    "Running": "green", "Succeeded": "green", "Ready": "green",
    "Pending": "orange", "Unknown": "gray",
    "Failed": "red", "NotReady": "red",
}


def _badge(status: str) -> str:
    return f":{_STATUS_COLOR.get(status, 'gray')}[●] {status}"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model" not in st.session_state:
    st.session_state.model = agent_core.GROQ_MODEL
if "current_context" not in st.session_state:
    st.session_state.current_context = None
if "current_namespace" not in st.session_state:
    st.session_state.current_namespace = None
if "chats" not in st.session_state:
    st.session_state.chats: dict[str, list[dict]] = {}
if "pending" not in st.session_state:
    st.session_state.pending: dict[str, agent_core.TurnResult] = {}
if "confirm_delete_ns" not in st.session_state:
    st.session_state.confirm_delete_ns = None

with st.sidebar:
    st.markdown("## ☸️ Kubernetes Agent")
    st.caption("Konwersacyjny agent do zarządzania klastrem")
    st.divider()
    st.subheader("Ustawienia")
    st.session_state.api_key = st.text_input(
        "Groq API key", type="password", value=st.session_state.api_key,
        help="Darmowy klucz: console.groq.com/keys",
    )
    st.session_state.model = st.text_input(
        "Model",
        value=st.session_state.model,
        help=(
            "Providery czasem retirują/zmieniają nazwy modeli. Jeśli dostajesz "
            "błąd 404 NOT_FOUND, sprawdź aktualną listę na "
            "console.groq.com/docs/models i wklej tu inny model."
        ),
    )

if st.session_state.api_key:
    agent_core.groq_client = groq.Groq(api_key=st.session_state.api_key)


def _error_message(exc: Exception) -> str:
    if isinstance(exc, UnicodeEncodeError):
        return (
            "Błąd kodowania znaków — Twoje środowisko (prawdopodobnie WSL2) nie ma "
            "ustawionego locale UTF-8. W terminalu WSL2, przed odpaleniem `streamlit run "
            "app.py`, wykonaj:\n\n```\nexport LANG=en_US.UTF-8\nexport PYTHONUTF8=1\n```"
        )
    if isinstance(exc, groq.RateLimitError):
        detail = ""
        if isinstance(exc.body, dict):
            detail = exc.body.get("error", {}).get("message", "")
        return (
            "⏳ Wyczerpany darmowy dzienny limit tokenów dla tego modelu na Groq.\n\n"
            f"{detail}\n\n"
            "Możesz albo poczekać do resetu limitu, albo od razu wpisać **inny model** w "
            "sidebarze (każdy model ma osobny limit) — np. `llama-3.1-8b-instant` lub "
            "`openai/gpt-oss-20b` — i kontynuować rozmowę bez czekania."
        )
    return f"Błąd: {exc}"


def _chat_key(context: str, namespace: str) -> str:
    # namespaces are only unique WITHIN a cluster - two different clusters
    # can both have a "default" namespace, so the key must include context.
    return f"{context}::{namespace}"


def _parse_args(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    # some models emit "null" instead of "{}" for zero-argument tools
    return parsed if isinstance(parsed, dict) else {}


def _render(msg: dict) -> None:
    role = msg.get("role")
    if role == "user":
        content = msg.get("content")
        if content:
            with st.chat_message("user"):
                st.write(content)
    elif role == "assistant":
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content")
        if tool_calls or content:
            with st.chat_message("assistant"):
                for tc in tool_calls:
                    fn = tc["function"]
                    args = _parse_args(fn["arguments"])
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    st.caption(f"🔧 `{fn['name']}({args_str})`")
                if content:
                    st.write(content)
    # role == "tool" -> skip, internal bookkeeping (tool result), not chat content


def _cluster_view() -> None:
    st.title("☸️ Klastry")
    st.caption("Kontekst kubeconfig = osobny klaster. Wybierz, żeby zobaczyć jego namespace'y.")
    try:
        contexts, active = agent_core.list_kube_contexts()
    except Exception as exc:
        st.error(f"Nie udało się odczytać ~/.kube/config: {exc}")
        return
    if not contexts:
        st.write("Brak skonfigurowanych klastrów w ~/.kube/config.")
        return
    cols = st.columns(3)
    for i, name in enumerate(contexts):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### 🖥️ {name}")
                if name == active:
                    st.caption(":green[●] aktualny kontekst kubectl")
                else:
                    st.caption(" ")
                if st.button("Otwórz →", key=f"open-ctx-{name}", use_container_width=True):
                    st.session_state.current_context = name
                    st.rerun()


def _namespace_view(context: str) -> None:
    if st.button("← Klastry"):
        st.session_state.current_context = None
        st.rerun()
    st.title(f"🧭 {context} — projekty (namespaces)")
    if not st.session_state.api_key:
        st.info("Wklej Groq API key w panelu bocznym, żeby móc czatować po wejściu w namespace.")

    with st.form(f"new-ns-{context}", clear_on_submit=True):
        c1, c2 = st.columns([4, 1])
        new_name = c1.text_input("Nowy namespace", label_visibility="collapsed", placeholder="nazwa-namespace")
        if c2.form_submit_button("+ Utwórz"):
            if new_name:
                st.toast(agent_core.create_namespace(new_name, context))
                st.rerun()
            else:
                st.warning("Podaj nazwę namespace'u.")

    confirm_target = st.session_state.confirm_delete_ns
    if confirm_target:
        with st.container(border=True):
            st.error(
                f"⚠️ Na pewno usunąć namespace **{confirm_target}**? To usunie WSZYSTKO "
                "w środku (pody, deploymenty, sekrety, wszystko) — nieodwracalnie."
            )
            cy, cn = st.columns(2)
            if cy.button("🗑️ Tak, usuń", key="confirm-del-ns", use_container_width=True):
                st.toast(agent_core.delete_namespace(confirm_target, context))
                st.session_state.confirm_delete_ns = None
                st.rerun()
            if cn.button("Anuluj", key="cancel-del-ns", use_container_width=True):
                st.session_state.confirm_delete_ns = None
                st.rerun()

    try:
        namespaces = agent_core.list_cluster_namespaces(context)
    except Exception as exc:
        st.error(f"Nie udało się pobrać namespace'ów z klastra: {exc}")
        return
    if not namespaces:
        st.write("Brak namespace'ów.")
        return
    cols = st.columns(3)
    for i, ns in enumerate(namespaces):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### 📁 {ns['name']}")
                st.caption(f"📦 {ns['pod_count']} pod(ów)")
                b1, b2 = st.columns([3, 1])
                if b1.button("Otwórz czat", key=f"open-ns-{context}-{ns['name']}", use_container_width=True):
                    st.session_state.current_namespace = ns["name"]
                    st.rerun()
                if b2.button("🗑️", key=f"del-ns-{context}-{ns['name']}", use_container_width=True):
                    st.session_state.confirm_delete_ns = ns["name"]
                    st.rerun()


def _chat_view(context: str, namespace: str) -> None:
    if st.button("← Namespaces"):
        st.session_state.current_namespace = None
        st.rerun()
    st.title(f"💬 {context} / {namespace}")

    try:
        pods = agent_core.list_namespace_pods(namespace, context)
        with st.expander(f"📦 Pody ({len(pods)})", expanded=bool(pods)):
            if pods:
                for p in pods:
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"`{p['name']}`")
                    c2.markdown(_badge(p["status"]))
            else:
                st.caption("Brak podów w tym namespace.")
    except Exception as exc:
        st.caption(f"Nie udało się pobrać podów: {exc}")

    if not st.session_state.api_key:
        st.info("Wklej Groq API key w panelu bocznym, żeby zacząć rozmowę.")
        return

    key = _chat_key(context, namespace)
    messages = st.session_state.chats.setdefault(key, [])
    tools = agent_core.build_tools(namespace, context)
    chat_config = agent_core.build_generate_config(namespace, tools)

    for msg in messages:
        _render(msg)

    pending = st.session_state.pending.get(key)
    if pending is not None:
        with st.container(border=True):
            st.error("⚠️ Agent chce wykonać poniższe akcje — potwierdź, żeby kontynuować:")
            for tc in pending.destructive_calls():
                args = _parse_args(tc.function.arguments)
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                st.markdown(f"🔧 `{tc.function.name}({args_str})`")
            col1, col2 = st.columns(2)
            approved = None
            if col1.button("✅ Zatwierdź", key=f"approve-{key}", use_container_width=True):
                approved = True
            if col2.button("❌ Odrzuć", key=f"reject-{key}", use_container_width=True):
                approved = False
            if approved is not None:
                result = None
                with st.spinner("Agent kontynuuje..."):
                    try:
                        result = agent_core.resume_turn(
                            messages, tools, chat_config, pending, approved, st.session_state.model
                        )
                    except Exception as exc:
                        messages.append({"role": "assistant", "content": _error_message(exc)})
                if result is not None and result.kind == "confirm":
                    st.session_state.pending[key] = result
                else:
                    st.session_state.pending.pop(key, None)
                st.rerun()
        return  # chat input stays hidden while a confirmation is pending

    user_input = st.chat_input("Napisz do agenta...")
    if user_input:
        messages.append({"role": "user", "content": user_input})
        result = None
        with st.spinner("Agent myśli..."):
            try:
                result = agent_core.run_turn(messages, tools, chat_config, st.session_state.model)
            except Exception as exc:
                messages.append({"role": "assistant", "content": _error_message(exc)})
        if result is not None and result.kind == "confirm":
            st.session_state.pending[key] = result
        st.rerun()


if st.session_state.current_context is None:
    _cluster_view()
elif st.session_state.current_namespace is None:
    _namespace_view(st.session_state.current_context)
else:
    _chat_view(st.session_state.current_context, st.session_state.current_namespace)
