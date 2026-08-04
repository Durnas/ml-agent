"""
agent_core.py - Conversational cluster-management agent, no UI.

Unlike the training-only agent this project started from, this one can
inspect and modify a whole Kubernetes namespace: pods, deployments,
services, jobs, events, GPU availability. Tools are scoped to ONE namespace
via a Python closure (build_tools(namespace)) - the LLM is never given a
namespace argument to pick from, so it cannot reach outside the namespace
its chat was opened for.

Destructive tools (anything in CONFIRM_TOOLS) don't execute immediately:
run_turn() stops and returns a "confirm" TurnResult instead, so a UI can
show the user what's about to happen and get an explicit go-ahead before
resume_turn() actually runs it.

LLM backend is Groq (OpenAI-compatible chat.completions API, genuinely free
tier). Tool schemas are inferred automatically from each tool function's type
hints + Google-style docstring (see _build_tool_schema) - the tool bodies
below don't know or care which LLM provider is calling them.

Requirements: pip install groq kubernetes
Env:          Kube config: ~/.kube/config (Minikube). GROQ_API_KEY is read
              by whichever UI sets `groq_client` (see app.py) - importing
              this module never touches the network or the cluster.
"""

import inspect
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import groq
from kubernetes import client, config

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Default only - the actual model used is chosen in the UI and passed into
# run_turn/resume_turn, since providers keep retiring/renaming model IDs
# (gemini-2.5-flash went 404-for-new-keys mid-project) and hardcoding one
# name means a single provider-side change breaks everyone using this code.
GROQ_MODEL = "openai/gpt-oss-120b"    # llama-3.3-70b-versatile deprecated by Groq, unreliable tool-calling near the end
TRAINER_IMAGE = "ml-trainer:latest"   # built inside Minikube's Docker daemon
MAX_TOOL_ITERATIONS = 8               # safety cap on tool calls within a single turn
MAX_LOG_TAIL_LINES = 500              # clamp on log-fetching tools
MAX_EVENTS = 20                       # clamp on get_recent_events
MAX_GENERATE_RETRIES = 3              # rate-limited / server overloaded - worth retrying
MAX_HISTORY_MESSAGES = 40             # cap on conversation history sent per API call - cost control
RETRY_BASE_DELAY_S = 2
RETRYABLE_ERRORS = (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError, groq.APITimeoutError)
_GITHUB_URL_RE = re.compile(r"^https://([^@/]+@)?github\.com/[^/]+/[^/]+/?$")

groq_client: groq.Groq  # set by the app once an API key is available


def _k8s(context: str | None = None) -> tuple[client.CoreV1Api, client.BatchV1Api, client.AppsV1Api]:
    """Loads kube config for the given kubeconfig context (None = whatever is
    marked current-context in ~/.kube/config) and returns the three API
    clients. Called lazily inside each tool so importing this module never
    touches the cluster."""
    config.load_kube_config(context=context)
    return client.CoreV1Api(), client.BatchV1Api(), client.AppsV1Api()


def _k8s_ext(context: str | None = None) -> dict:
    """Same idea as _k8s() but for the less-frequently-needed API groups
    (Ingress/NetworkPolicy, StorageClass, RBAC, HPA, metrics.k8s.io), kept
    separate so the original three-client tuple stays unchanged for every
    existing tool/test."""
    config.load_kube_config(context=context)
    return {
        "networking": client.NetworkingV1Api(),
        "storage": client.StorageV1Api(),
        "rbac": client.RbacAuthorizationV1Api(),
        "autoscaling": client.AutoscalingV1Api(),
        "metrics": client.CustomObjectsApi(),
        "policy": client.PolicyV1Api(),
        "apiextensions": client.ApiextensionsV1Api(),
    }


def list_kube_contexts() -> tuple[list[str], str | None]:
    """Returns (all context names, current-context name) from ~/.kube/config -
    one context per Minikube profile / cluster you've configured. Used by the
    dashboard's top-level cluster picker."""
    contexts, active = config.list_kube_config_contexts()
    names = [c["name"] for c in contexts]
    return names, (active["name"] if active else None)


def list_cluster_namespaces(context: str | None = None) -> list[dict]:
    """Returns [{"name": ..., "pod_count": ...}, ...] for the dashboard's
    namespace list. Not an LLM tool - plain cluster read used directly by the UI."""
    core, _, _ = _k8s(context)
    namespaces = core.list_namespace().items
    result = []
    for ns in namespaces:
        pods = core.list_namespaced_pod(namespace=ns.metadata.name).items
        result.append({"name": ns.metadata.name, "pod_count": len(pods)})
    return result


def list_namespace_pods(namespace: str, context: str | None = None) -> list[dict]:
    """Returns [{"name": ..., "status": ...}, ...] for the live pod panel shown
    when a namespace's chat is opened. Not an LLM tool - plain cluster read."""
    core, _, _ = _k8s(context)
    pods = core.list_namespaced_pod(namespace=namespace).items
    return [{"name": p.metadata.name, "status": p.status.phase} for p in pods]


def create_namespace(name: str, context: str | None = None) -> str:
    """Creates a namespace. Not an LLM tool - the dashboard's namespace list is
    the only place a namespace can be created, as a direct UI action (no chat,
    no LLM in the loop) since it doesn't need natural-language flexibility."""
    core, _, _ = _k8s(context)
    try:
        core.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=name)))
    except client.exceptions.ApiException as exc:
        return f"Could not create namespace '{name}': {exc.reason}"
    return f"Namespace '{name}' created."


def delete_namespace(name: str, context: str | None = None) -> str:
    """Deletes a namespace and everything in it. Not an LLM tool - same reasoning
    as create_namespace, plus this is too destructive to ever trust to an LLM's
    interpretation of intent; the dashboard gates it behind its own explicit
    two-step UI confirmation before this is called."""
    core, _, _ = _k8s(context)
    try:
        core.delete_namespace(name=name)
    except client.exceptions.ApiException as exc:
        return f"Could not delete namespace '{name}': {exc.reason}"
    return f"Namespace '{name}' deleted."


# ----------------------------------------------------------------------------
# Tool factory - everything below is scoped to one namespace, closed over so
# the LLM can never pick a different one mid-conversation.
# ----------------------------------------------------------------------------
def build_tools(namespace: str, context: str | None = None) -> dict[str, callable]:
    def list_pods() -> str:
        """Lists all pods in this namespace with their phase (Running, Pending, etc.)."""
        core, _, _ = _k8s(context)
        pods = core.list_namespaced_pod(namespace=namespace).items
        if not pods:
            return "No pods found."
        return "\n".join(f"- {p.metadata.name}: {p.status.phase}" for p in pods)

    def list_deployments() -> str:
        """Lists all Deployments in this namespace with ready/desired replica counts."""
        _, _, apps = _k8s(context)
        deployments = apps.list_namespaced_deployment(namespace=namespace).items
        if not deployments:
            return "No deployments found."
        lines = []
        for d in deployments:
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            lines.append(f"- {d.metadata.name}: {ready}/{desired} ready")
        return "\n".join(lines)

    def list_services() -> str:
        """Lists all Services in this namespace with their type and ports."""
        core, _, _ = _k8s(context)
        services = core.list_namespaced_service(namespace=namespace).items
        if not services:
            return "No services found."
        lines = []
        for s in services:
            ports = ",".join(str(p.port) for p in (s.spec.ports or []))
            lines.append(f"- {s.metadata.name}: type={s.spec.type}, ports={ports}")
        return "\n".join(lines)

    def list_jobs() -> str:
        """Lists all training Jobs in this namespace, with their status."""
        _, batch, _ = _k8s(context)
        jobs = batch.list_namespaced_job(namespace=namespace).items
        if not jobs:
            return "No jobs found."
        lines = []
        for j in jobs:
            if j.status.succeeded:
                state = "Succeeded"
            elif j.status.failed:
                state = "Failed"
            elif j.status.active:
                state = "Running"
            else:
                state = "Pending"
            lines.append(f"- {j.metadata.name}: {state}")
        return "\n".join(lines)

    def get_job_status(job_name: str) -> str:
        """Returns the current status of a specific training Job.

        Args:
            job_name: Name of the Job, as returned by create_training_job or list_jobs.
        """
        _, batch, _ = _k8s(context)
        try:
            job = batch.read_namespaced_job(name=job_name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not read job '{job_name}': {exc.reason}"
        if job.status.succeeded:
            return f"Job '{job_name}': Succeeded"
        if job.status.failed:
            return f"Job '{job_name}': Failed"
        if job.status.active:
            return f"Job '{job_name}': Running"
        return f"Job '{job_name}': Pending"

    def get_job_logs(job_name: str, tail_lines: int = 50) -> str:
        """Fetches a recent-logs snapshot (not a live stream) from a training Job's pod.

        Args:
            job_name: Name of the Job to fetch logs for.
            tail_lines: How many of the most recent log lines to return.
        """
        clamped = max(1, min(tail_lines, MAX_LOG_TAIL_LINES))
        core, _, _ = _k8s(context)
        pods = core.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}"
        ).items
        if not pods:
            return f"No pod found for job '{job_name}' yet."
        pod_name = pods[0].metadata.name
        try:
            return core.read_namespaced_pod_log(
                name=pod_name, namespace=namespace, tail_lines=clamped
            )
        except client.exceptions.ApiException as exc:
            return f"Could not read logs for pod '{pod_name}': {exc.reason}"

    def get_pod_logs(pod_name: str, tail_lines: int = 50) -> str:
        """Fetches a recent-logs snapshot (not a live stream) from any pod by name.

        Args:
            pod_name: Name of the pod to fetch logs for.
            tail_lines: How many of the most recent log lines to return.
        """
        clamped = max(1, min(tail_lines, MAX_LOG_TAIL_LINES))
        core, _, _ = _k8s(context)
        try:
            return core.read_namespaced_pod_log(
                name=pod_name, namespace=namespace, tail_lines=clamped
            )
        except client.exceptions.ApiException as exc:
            return f"Could not read logs for pod '{pod_name}': {exc.reason}"

    def get_recent_events() -> str:
        """Lists the most recent Kubernetes events in this namespace (useful for
        diagnosing crashes, scheduling failures, image pull errors, etc.)."""
        core, _, _ = _k8s(context)
        events = core.list_namespaced_event(namespace=namespace).items
        if not events:
            return "No recent events."
        events = sorted(
            events, key=lambda e: e.last_timestamp or e.event_time or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )[:MAX_EVENTS]
        return "\n".join(
            f"- [{e.type}] {e.involved_object.kind}/{e.involved_object.name}: {e.reason} - {e.message}"
            for e in events
        )

    def check_gpu_availability() -> str:
        """Checks how many GPUs are allocatable on each cluster node."""
        core, _, _ = _k8s(context)
        nodes = core.list_node().items
        if not nodes:
            return "No nodes found."
        lines = []
        for n in nodes:
            gpus = (n.status.allocatable or {}).get("nvidia.com/gpu", "0")
            lines.append(f"- {n.metadata.name}: {gpus} GPU(s) allocatable")
        return "\n".join(lines)

    def create_training_job(github_url: str) -> str:
        """Creates a new GPU training Job from a GitHub repo in this namespace and
        returns immediately without waiting for it to finish. The repo must
        contain train.py in its root.

        Args:
            github_url: Full GitHub repository URL, e.g. https://github.com/user/repo
        """
        if not _GITHUB_URL_RE.match(github_url.strip()):
            return (
                f"'{github_url}' is not a valid GitHub repository URL. Expected format: "
                "https://github.com/user/repo (or https://<TOKEN>@github.com/user/repo for "
                "private repos). Ask the user for the real link - do not guess or invent one."
            )
        _, batch, _ = _k8s(context)
        job_name = f"train-job-{uuid.uuid4().hex[:8]}"
        container = client.V1Container(
            name="trainer",
            image=TRAINER_IMAGE,
            image_pull_policy="Never",
            env=[client.V1EnvVar(name="GITHUB_URL", value=github_url)],
            resources=client.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
        )
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                backoff_limit=0,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"job-name": job_name}),
                    spec=client.V1PodSpec(containers=[container], restart_policy="Never"),
                ),
            ),
        )
        batch.create_namespaced_job(namespace=namespace, body=job)
        return (
            f"Created job '{job_name}' for {github_url}. "
            "Use get_job_status or get_job_logs to check on it."
        )

    def delete_job(job_name: str) -> str:
        """Deletes a training Job and its pods.

        Args:
            job_name: Name of the Job to delete.
        """
        _, batch, _ = _k8s(context)
        try:
            batch.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy="Foreground"),
            )
        except client.exceptions.ApiException as exc:
            return f"Could not delete job '{job_name}': {exc.reason}"
        return f"Job '{job_name}' deleted."

    def delete_pod(pod_name: str) -> str:
        """Deletes a single pod by name.

        Args:
            pod_name: Name of the pod to delete.
        """
        core, _, _ = _k8s(context)
        try:
            core.delete_namespaced_pod(name=pod_name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete pod '{pod_name}': {exc.reason}"
        return f"Pod '{pod_name}' deleted."

    def scale_deployment(deployment_name: str, replicas: int) -> str:
        """Scales a Deployment to a given number of replicas.

        Args:
            deployment_name: Name of the Deployment to scale.
            replicas: Target replica count.
        """
        _, _, apps = _k8s(context)
        try:
            apps.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body={"spec": {"replicas": replicas}},
            )
        except client.exceptions.ApiException as exc:
            return f"Could not scale deployment '{deployment_name}': {exc.reason}"
        return f"Deployment '{deployment_name}' scaled to {replicas} replicas."

    def restart_deployment(deployment_name: str) -> str:
        """Triggers a rolling restart of a Deployment (like `kubectl rollout restart`).

        Args:
            deployment_name: Name of the Deployment to restart.
        """
        _, _, apps = _k8s(context)
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()
                        }
                    }
                }
            }
        }
        try:
            apps.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=patch)
        except client.exceptions.ApiException as exc:
            return f"Could not restart deployment '{deployment_name}': {exc.reason}"
        return f"Deployment '{deployment_name}' restart triggered."

    def delete_deployment(deployment_name: str) -> str:
        """Deletes a Deployment and all its pods.

        Args:
            deployment_name: Name of the Deployment to delete.
        """
        _, _, apps = _k8s(context)
        try:
            apps.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete deployment '{deployment_name}': {exc.reason}"
        return f"Deployment '{deployment_name}' deleted."

    def create_deployment(name: str, image: str, replicas: int = 1, container_port: int = 0) -> str:
        """Creates a general-purpose Deployment running any container image -
        nginx, redis, a custom app, anything - not just ML training (that's
        what create_training_job is for, and it requires a GPU). Use this to
        run ordinary workloads, or to spin up something disposable to test
        scale_deployment/restart_deployment/delete_pod against.

        Args:
            name: Name for the Deployment.
            image: Container image to run, e.g. nginx or redis:7.
            replicas: Number of replicas to start.
            container_port: Port the container listens on, if any (0 = none).
        """
        _, _, apps = _k8s(context)
        container = client.V1Container(
            name=name,
            image=image,
            ports=[client.V1ContainerPort(container_port=container_port)] if container_port else None,
        )
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(match_labels={"app": name}),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"app": name}),
                    spec=client.V1PodSpec(containers=[container]),
                ),
            ),
        )
        try:
            apps.create_namespaced_deployment(namespace=namespace, body=deployment)
        except client.exceptions.ApiException as exc:
            return f"Could not create deployment '{name}': {exc.reason}"
        return f"Created deployment '{name}' ({replicas} replica(s) of {image})."

    def list_configmaps() -> str:
        """Lists all ConfigMaps in this namespace."""
        core, _, _ = _k8s(context)
        cms = core.list_namespaced_config_map(namespace=namespace).items
        if not cms:
            return "No ConfigMaps found."
        return "\n".join(f"- {c.metadata.name}" for c in cms)

    def get_configmap(name: str) -> str:
        """Returns the key/value data of a ConfigMap.

        Args:
            name: Name of the ConfigMap.
        """
        core, _, _ = _k8s(context)
        try:
            cm = core.read_namespaced_config_map(name=name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not read ConfigMap '{name}': {exc.reason}"
        data = cm.data or {}
        if not data:
            return f"ConfigMap '{name}' has no data."
        return "\n".join(f"- {k}: {v}" for k, v in data.items())

    def delete_configmap(name: str) -> str:
        """Deletes a ConfigMap.

        Args:
            name: Name of the ConfigMap to delete.
        """
        core, _, _ = _k8s(context)
        try:
            core.delete_namespaced_config_map(name=name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete ConfigMap '{name}': {exc.reason}"
        return f"ConfigMap '{name}' deleted."

    def list_secrets() -> str:
        """Lists all Secrets in this namespace by name and type only. Secret
        VALUES are never read or exposed to the LLM, for security."""
        core, _, _ = _k8s(context)
        secrets = core.list_namespaced_secret(namespace=namespace).items
        if not secrets:
            return "No Secrets found."
        return "\n".join(f"- {s.metadata.name} ({s.type})" for s in secrets)

    def delete_secret(name: str) -> str:
        """Deletes a Secret.

        Args:
            name: Name of the Secret to delete.
        """
        core, _, _ = _k8s(context)
        try:
            core.delete_namespaced_secret(name=name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete Secret '{name}': {exc.reason}"
        return f"Secret '{name}' deleted."

    def list_pvcs() -> str:
        """Lists all PersistentVolumeClaims in this namespace with status and size."""
        core, _, _ = _k8s(context)
        pvcs = core.list_namespaced_persistent_volume_claim(namespace=namespace).items
        if not pvcs:
            return "No PersistentVolumeClaims found."
        lines = []
        for p in pvcs:
            size = (p.status.capacity or {}).get("storage", "?")
            lines.append(f"- {p.metadata.name}: {p.status.phase}, {size}")
        return "\n".join(lines)

    def delete_pvc(name: str) -> str:
        """Deletes a PersistentVolumeClaim. Depending on the StorageClass's
        reclaim policy this may also delete the underlying stored data.

        Args:
            name: Name of the PersistentVolumeClaim to delete.
        """
        core, _, _ = _k8s(context)
        try:
            core.delete_namespaced_persistent_volume_claim(name=name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete PVC '{name}': {exc.reason}"
        return f"PVC '{name}' deleted."

    def list_storage_classes() -> str:
        """Lists all StorageClasses available in the cluster (not namespace-scoped)."""
        ext = _k8s_ext(context)
        scs = ext["storage"].list_storage_class().items
        if not scs:
            return "No StorageClasses found."
        return "\n".join(f"- {s.metadata.name} (provisioner: {s.provisioner})" for s in scs)

    def list_ingresses() -> str:
        """Lists all Ingresses in this namespace with their hosts."""
        ext = _k8s_ext(context)
        ingresses = ext["networking"].list_namespaced_ingress(namespace=namespace).items
        if not ingresses:
            return "No Ingresses found."
        lines = []
        for i in ingresses:
            hosts = ",".join(r.host or "*" for r in (i.spec.rules or []))
            lines.append(f"- {i.metadata.name}: {hosts}")
        return "\n".join(lines)

    def delete_ingress(name: str) -> str:
        """Deletes an Ingress.

        Args:
            name: Name of the Ingress to delete.
        """
        ext = _k8s_ext(context)
        try:
            ext["networking"].delete_namespaced_ingress(name=name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not delete Ingress '{name}': {exc.reason}"
        return f"Ingress '{name}' deleted."

    def list_network_policies() -> str:
        """Lists all NetworkPolicies in this namespace."""
        ext = _k8s_ext(context)
        policies = ext["networking"].list_namespaced_network_policy(namespace=namespace).items
        if not policies:
            return "No NetworkPolicies found."
        return "\n".join(f"- {p.metadata.name}" for p in policies)

    def list_service_accounts() -> str:
        """Lists all ServiceAccounts in this namespace."""
        core, _, _ = _k8s(context)
        accounts = core.list_namespaced_service_account(namespace=namespace).items
        if not accounts:
            return "No ServiceAccounts found."
        return "\n".join(f"- {a.metadata.name}" for a in accounts)

    def list_roles() -> str:
        """Lists all Roles (RBAC) in this namespace. Read-only - RBAC objects
        are not created/modified by this agent, to avoid misconfiguring
        cluster permissions or security."""
        ext = _k8s_ext(context)
        roles = ext["rbac"].list_namespaced_role(namespace=namespace).items
        if not roles:
            return "No Roles found."
        return "\n".join(f"- {r.metadata.name}" for r in roles)

    def list_role_bindings() -> str:
        """Lists all RoleBindings (RBAC) in this namespace, with the role and
        subjects each one binds."""
        ext = _k8s_ext(context)
        bindings = ext["rbac"].list_namespaced_role_binding(namespace=namespace).items
        if not bindings:
            return "No RoleBindings found."
        lines = []
        for b in bindings:
            subjects = ",".join(s.name for s in (b.subjects or []))
            lines.append(f"- {b.metadata.name}: role={b.role_ref.name}, subjects=[{subjects}]")
        return "\n".join(lines)

    def list_statefulsets() -> str:
        """Lists all StatefulSets in this namespace with ready/desired replica counts."""
        _, _, apps = _k8s(context)
        sets = apps.list_namespaced_stateful_set(namespace=namespace).items
        if not sets:
            return "No StatefulSets found."
        lines = []
        for s in sets:
            ready = s.status.ready_replicas or 0
            desired = s.spec.replicas or 0
            lines.append(f"- {s.metadata.name}: {ready}/{desired} ready")
        return "\n".join(lines)

    def scale_statefulset(name: str, replicas: int) -> str:
        """Scales a StatefulSet to a given number of replicas.

        Args:
            name: Name of the StatefulSet to scale.
            replicas: Target replica count.
        """
        _, _, apps = _k8s(context)
        try:
            apps.patch_namespaced_stateful_set_scale(
                name=name, namespace=namespace, body={"spec": {"replicas": replicas}}
            )
        except client.exceptions.ApiException as exc:
            return f"Could not scale StatefulSet '{name}': {exc.reason}"
        return f"StatefulSet '{name}' scaled to {replicas} replicas."

    def restart_statefulset(name: str) -> str:
        """Triggers a rolling restart of a StatefulSet.

        Args:
            name: Name of the StatefulSet to restart.
        """
        _, _, apps = _k8s(context)
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()
                        }
                    }
                }
            }
        }
        try:
            apps.patch_namespaced_stateful_set(name=name, namespace=namespace, body=patch)
        except client.exceptions.ApiException as exc:
            return f"Could not restart StatefulSet '{name}': {exc.reason}"
        return f"StatefulSet '{name}' restart triggered."

    def list_daemonsets() -> str:
        """Lists all DaemonSets in this namespace with ready/desired pod counts."""
        _, _, apps = _k8s(context)
        sets = apps.list_namespaced_daemon_set(namespace=namespace).items
        if not sets:
            return "No DaemonSets found."
        lines = []
        for s in sets:
            ready = s.status.number_ready or 0
            desired = s.status.desired_number_scheduled or 0
            lines.append(f"- {s.metadata.name}: {ready}/{desired} ready")
        return "\n".join(lines)

    def restart_daemonset(name: str) -> str:
        """Triggers a rolling restart of a DaemonSet.

        Args:
            name: Name of the DaemonSet to restart.
        """
        _, _, apps = _k8s(context)
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()
                        }
                    }
                }
            }
        }
        try:
            apps.patch_namespaced_daemon_set(name=name, namespace=namespace, body=patch)
        except client.exceptions.ApiException as exc:
            return f"Could not restart DaemonSet '{name}': {exc.reason}"
        return f"DaemonSet '{name}' restart triggered."

    def list_hpas() -> str:
        """Lists all HorizontalPodAutoscalers in this namespace with current/min/max replicas."""
        ext = _k8s_ext(context)
        hpas = ext["autoscaling"].list_namespaced_horizontal_pod_autoscaler(namespace=namespace).items
        if not hpas:
            return "No HorizontalPodAutoscalers found."
        lines = []
        for h in hpas:
            lines.append(
                f"- {h.metadata.name}: current={h.status.current_replicas}, "
                f"min={h.spec.min_replicas}, max={h.spec.max_replicas}"
            )
        return "\n".join(lines)

    def list_nodes() -> str:
        """Lists all nodes in the CLUSTER (not namespace-scoped - nodes aren't
        namespaced objects in Kubernetes) with their Ready/schedulable status."""
        core, _, _ = _k8s(context)
        nodes = core.list_node().items
        if not nodes:
            return "No nodes found."
        lines = []
        for n in nodes:
            ready = "Unknown"
            for cond in n.status.conditions or []:
                if cond.type == "Ready":
                    ready = "Ready" if cond.status == "True" else "NotReady"
            schedulable = "SchedulingDisabled" if n.spec.unschedulable else "Schedulable"
            lines.append(f"- {n.metadata.name}: {ready}, {schedulable}")
        return "\n".join(lines)

    def cordon_node(node_name: str) -> str:
        """Marks a node as unschedulable - existing pods keep running, but no
        new pods will be scheduled onto it. CLUSTER-WIDE, not namespace-scoped.
        Controls scheduling ELIGIBILITY only - it does not free or add any
        CPU/memory/GPU capacity. Cordoning (or draining, or uncordoning) a
        node can NEVER fix a "Pending" pod caused by insufficient resources -
        do not try it for that. It only helps when the node itself is
        unhealthy and you want to stop it from receiving new work.

        Args:
            node_name: Name of the node to cordon.
        """
        core, _, _ = _k8s(context)
        try:
            core.patch_node(name=node_name, body={"spec": {"unschedulable": True}})
        except client.exceptions.ApiException as exc:
            return f"Could not cordon node '{node_name}': {exc.reason}"
        return f"Node '{node_name}' cordoned."

    def uncordon_node(node_name: str) -> str:
        """Marks a node as schedulable again. CLUSTER-WIDE, not namespace-scoped.
        This does not add any CPU/memory/GPU capacity - it only undoes a
        previous cordon. It cannot fix a "Pending" pod caused by insufficient
        resources.

        Args:
            node_name: Name of the node to uncordon.
        """
        core, _, _ = _k8s(context)
        try:
            core.patch_node(name=node_name, body={"spec": {"unschedulable": False}})
        except client.exceptions.ApiException as exc:
            return f"Could not uncordon node '{node_name}': {exc.reason}"
        return f"Node '{node_name}' uncordoned."

    def drain_node(node_name: str) -> str:
        """Cordons a node and evicts all its evictable pods (like `kubectl
        drain`), respecting PodDisruptionBudgets. DaemonSet-managed pods are
        left running since they can't be rescheduled elsewhere. Stronger than
        cordon_node - use this to actually empty a node, not just stop new
        scheduling onto it. CLUSTER-WIDE, not namespace-scoped. Does NOT free
        or add any CPU/memory/GPU capacity - evicted pods go right back to
        Pending if nothing else in the cluster can fit them. Never use this
        to try to fix an "insufficient resources" scheduling failure; it has
        no effect on that.

        Args:
            node_name: Name of the node to drain.
        """
        core, _, _ = _k8s(context)
        try:
            core.patch_node(name=node_name, body={"spec": {"unschedulable": True}})
        except client.exceptions.ApiException as exc:
            return f"Could not cordon node '{node_name}' before draining: {exc.reason}"

        try:
            pods = core.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}").items
        except client.exceptions.ApiException as exc:
            return f"Node '{node_name}' cordoned, but could not list its pods: {exc.reason}"

        evicted, skipped, failed = [], [], []
        for pod in pods:
            if any(o.kind == "DaemonSet" for o in (pod.metadata.owner_references or [])):
                skipped.append(pod.metadata.name)
                continue
            eviction = client.V1Eviction(
                metadata=client.V1ObjectMeta(name=pod.metadata.name, namespace=pod.metadata.namespace)
            )
            try:
                core.create_namespaced_pod_eviction(
                    name=pod.metadata.name, namespace=pod.metadata.namespace, body=eviction
                )
                evicted.append(pod.metadata.name)
            except client.exceptions.ApiException as exc:
                failed.append(f"{pod.metadata.name} ({exc.reason})")

        parts = [f"Node '{node_name}' cordoned."]
        if evicted:
            parts.append(f"Evicted: {', '.join(evicted)}.")
        if skipped:
            parts.append(f"Left running (DaemonSet-managed): {', '.join(skipped)}.")
        if failed:
            parts.append(f"Failed to evict (likely blocked by a PodDisruptionBudget): {', '.join(failed)}.")
        if not pods:
            parts.append("No pods were running on it.")
        return " ".join(parts)

    def list_resource_quotas() -> str:
        """Lists ResourceQuotas in this namespace with used vs. hard limits -
        explains why creating a pod/resource might be rejected for exceeding quota."""
        core, _, _ = _k8s(context)
        quotas = core.list_namespaced_resource_quota(namespace=namespace).items
        if not quotas:
            return "No ResourceQuotas found."
        lines = []
        for q in quotas:
            used = dict(q.status.used or {})
            hard = dict(q.status.hard or {})
            lines.append(f"- {q.metadata.name}: used={used}, hard={hard}")
        return "\n".join(lines)

    def list_limit_ranges() -> str:
        """Lists LimitRanges in this namespace - default/min/max resource
        constraints automatically applied to new pods/containers."""
        core, _, _ = _k8s(context)
        ranges = core.list_namespaced_limit_range(namespace=namespace).items
        if not ranges:
            return "No LimitRanges found."
        lines = []
        for lr in ranges:
            items = [
                f"{it.type}: max={dict(it.max or {})}, min={dict(it.min or {})}, default={dict(it.default or {})}"
                for it in (lr.spec.limits or [])
            ]
            lines.append(f"- {lr.metadata.name}: " + "; ".join(items))
        return "\n".join(lines)

    def list_pod_disruption_budgets() -> str:
        """Lists PodDisruptionBudgets in this namespace with allowed/current/
        desired healthy replica counts - relevant when scaling, draining, or
        restarting, since a PDB can block those operations."""
        ext = _k8s_ext(context)
        pdbs = ext["policy"].list_namespaced_pod_disruption_budget(namespace=namespace).items
        if not pdbs:
            return "No PodDisruptionBudgets found."
        lines = []
        for p in pdbs:
            lines.append(
                f"- {p.metadata.name}: allowed_disruptions={p.status.disruptions_allowed}, "
                f"current_healthy={p.status.current_healthy}, desired_healthy={p.status.desired_healthy}"
            )
        return "\n".join(lines)

    def list_crds() -> str:
        """Lists CustomResourceDefinitions installed in the CLUSTER (not
        namespace-scoped) - shows what operators/extensions are available."""
        ext = _k8s_ext(context)
        crds = ext["apiextensions"].list_custom_resource_definition().items
        if not crds:
            return "No CustomResourceDefinitions found."
        return "\n".join(f"- {c.metadata.name}" for c in crds)

    def describe_pod(pod_name: str) -> str:
        """Returns detailed diagnostic info for a pod: phase, and per-container
        state including restart count and the reason for the last crash/wait
        (e.g. CrashLoopBackOff, OOMKilled, ImagePullBackOff), plus resource
        requests/limits. This is the primary tool for diagnosing a broken pod -
        equivalent to `kubectl describe pod`. Prefer this over guessing from
        logs alone when a pod isn't Running.

        Args:
            pod_name: Name of the pod to describe.
        """
        core, _, _ = _k8s(context)
        try:
            pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
        except client.exceptions.ApiException as exc:
            return f"Could not read pod '{pod_name}': {exc.reason}"
        lines = [f"Phase: {pod.status.phase}"]
        # A pod stuck in Pending usually hasn't been scheduled onto any node
        # yet, so container_statuses below is empty - the real reason (e.g.
        # "Insufficient nvidia.com/gpu") lives in the PodScheduled condition,
        # not per-container state. Surface it first since it's the most
        # common real-world cause of a stuck pod.
        for cond in pod.status.conditions or []:
            if cond.type == "PodScheduled" and cond.status != "True":
                lines.append(f"Not scheduled ({cond.reason}): {cond.message}")
        if not pod.status.container_statuses:
            lines.append("No containers have started yet (pod is not yet scheduled onto a node).")
        for cs in pod.status.container_statuses or []:
            state = cs.state
            if state.waiting:
                state_str = f"Waiting ({state.waiting.reason}: {state.waiting.message})"
            elif state.terminated:
                state_str = f"Terminated (exit_code={state.terminated.exit_code}, reason={state.terminated.reason})"
            elif state.running:
                state_str = "Running"
            else:
                state_str = "Unknown"
            limits = dict((cs.resources.limits if cs.resources else None) or {})
            requests = dict((cs.resources.requests if cs.resources else None) or {})
            lines.append(
                f"Container '{cs.name}': ready={cs.ready}, restarts={cs.restart_count}, "
                f"state={state_str}, requests={requests}, limits={limits}"
            )
        return "\n".join(lines)

    def get_resource_usage() -> str:
        """Returns live CPU/memory usage per pod in this namespace, from the
        metrics-server addon. If metrics-server isn't installed in the
        cluster, says so clearly instead of erroring (common on a fresh
        Minikube - enable with `minikube addons enable metrics-server`)."""
        ext = _k8s_ext(context)
        try:
            data = ext["metrics"].list_namespaced_custom_object(
                group="metrics.k8s.io", version="v1beta1", namespace=namespace, plural="pods"
            )
        except client.exceptions.ApiException as exc:
            if exc.status == 404:
                return (
                    "metrics-server is not installed in this cluster (needed for live "
                    "resource usage). Enable it with: minikube addons enable metrics-server"
                )
            return f"Could not fetch resource usage: {exc.reason}"
        items = data.get("items", [])
        if not items:
            return "No metrics available yet."
        lines = []
        for item in items:
            pod_name = item["metadata"]["name"]
            for c in item.get("containers", []):
                usage = c.get("usage", {})
                lines.append(f"- {pod_name}/{c['name']}: cpu={usage.get('cpu')}, memory={usage.get('memory')}")
        return "\n".join(lines)

    def check_service_endpoints(service_name: str) -> str:
        """Checks whether a Service's selector actually matches any running
        pods - diagnoses the classic 'traffic goes nowhere' bug caused by a
        label mismatch between the Service and its target pods.

        Args:
            service_name: Name of the Service to check.
        """
        core, _, _ = _k8s(context)
        try:
            endpoints = core.list_namespaced_endpoints(
                namespace=namespace, field_selector=f"metadata.name={service_name}"
            ).items
        except client.exceptions.ApiException as exc:
            return f"Could not read endpoints for '{service_name}': {exc.reason}"
        if not endpoints:
            return f"No Endpoints object found for service '{service_name}' - does the service exist?"
        addresses = [
            addr.ip for subset in (endpoints[0].subsets or []) for addr in (subset.addresses or [])
        ]
        if not addresses:
            return (
                f"Service '{service_name}' has NO matching pod endpoints - its selector "
                "doesn't match any running pod's labels, so traffic to it will fail. "
                "Check the Service's selector against the target pods' labels."
            )
        return f"Service '{service_name}' has {len(addresses)} healthy endpoint(s): {', '.join(addresses)}"

    return {
        fn.__name__: fn
        for fn in (
            list_pods,
            list_deployments,
            list_services,
            list_jobs,
            get_job_status,
            get_job_logs,
            get_pod_logs,
            get_recent_events,
            check_gpu_availability,
            create_training_job,
            delete_job,
            delete_pod,
            scale_deployment,
            restart_deployment,
            create_deployment,
            delete_deployment,
            list_configmaps,
            get_configmap,
            delete_configmap,
            list_secrets,
            delete_secret,
            list_pvcs,
            delete_pvc,
            list_storage_classes,
            list_ingresses,
            delete_ingress,
            list_network_policies,
            list_service_accounts,
            list_roles,
            list_role_bindings,
            list_statefulsets,
            scale_statefulset,
            restart_statefulset,
            list_daemonsets,
            restart_daemonset,
            list_hpas,
            list_nodes,
            cordon_node,
            uncordon_node,
            drain_node,
            list_resource_quotas,
            list_limit_ranges,
            list_pod_disruption_budgets,
            list_crds,
            describe_pod,
            get_resource_usage,
            check_service_endpoints,
        )
    }


# Tools in this set pause for user confirmation instead of executing immediately.
CONFIRM_TOOLS = {
    "create_training_job",
    "delete_job",
    "delete_pod",
    "scale_deployment",
    "restart_deployment",
    "create_deployment",
    "delete_deployment",
    "delete_configmap",
    "delete_secret",
    "delete_pvc",
    "delete_ingress",
    "scale_statefulset",
    "restart_statefulset",
    "restart_daemonset",
    "cordon_node",
    "uncordon_node",
    "drain_node",
}


_JSON_TYPE_BY_PYTHON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Splits a Google-style docstring into (summary, {param_name: description})."""
    lines = (doc or "").strip().split("\n")
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line == "Args:":
            in_args = True
            continue
        if in_args:
            match = re.match(r"(\w+):\s*(.*)", line)
            if match:
                current_param = match.group(1)
                param_docs[current_param] = match.group(2)
            elif current_param and line:
                param_docs[current_param] += " " + line
        elif line:
            summary_lines.append(line)
    return " ".join(summary_lines), param_docs


def _build_tool_schema(fn) -> dict:
    """Builds an OpenAI/Groq-compatible tool schema from a plain Python
    function's signature and docstring - the tool functions in build_tools()
    don't need to know or care about this; adding a new tool there is
    automatically picked up here with zero extra schema-writing."""
    summary, param_docs = _parse_docstring(fn.__doc__)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        py_type = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[name] = {
            "type": _JSON_TYPE_BY_PYTHON_TYPE.get(py_type, "string"),
            "description": param_docs.get(name, ""),
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": summary,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


@dataclass
class ChatConfig:
    system_prompt: str
    tool_schemas: list[dict]


def build_generate_config(namespace: str, tools: dict[str, callable]) -> ChatConfig:
    system_prompt = (
        f"You are an MLOps/DevOps assistant managing the Kubernetes namespace '{namespace}'. "
        "You can inspect and modify pods, deployments, statefulsets, daemonsets, services, "
        "ingresses, network policies, jobs, configmaps, secrets (names/types only, never "
        "values), PVCs, storage classes, RBAC (service accounts/roles/role bindings, "
        "read-only), horizontal pod autoscalers, resource quotas, limit ranges, pod "
        "disruption budgets, custom resource definitions, cluster nodes (including "
        "draining), and GPU availability. You can launch GPU training jobs from a GitHub "
        "repo (it must contain train.py in its root) via create_training_job, or launch "
        "ordinary (non-GPU) workloads via create_deployment for anything else. Namespaces "
        "themselves are created/deleted from the dashboard's namespace list directly, "
        "not through this chat. "
        "You have NO shell, terminal, SSH, or CLI access of any kind - not to minikube, "
        "kubectl, docker, or anything else. The tool functions listed are the ONLY actions "
        "you can take, period. NEVER say you will 'run', 'execute', or 'enable' a command "
        "like `minikube start ...` or any other shell command, and never ask the user for "
        "permission to run one - you physically cannot. If fixing something requires an "
        "action outside your tools (changing Minikube's startup flags, installing a driver, "
        "restarting the cluster), say so plainly and tell the user the exact command to run "
        "THEMSELVES in their own terminal - don't imply you'll do it. "
        "Modifying/deleting actions always require the user's explicit go-ahead before they "
        "run - propose them and wait, don't assume approval. "
        "NEVER call a tool with a placeholder, example, or made-up value for a required "
        "argument (e.g. a fake GitHub URL like 'paste_link_here' or a guessed pod/job name) "
        "just to have something to fill the field with. If you don't have a real value the "
        "user actually gave you, ask them for it in plain text instead of calling the tool. "
        "Node operations (cordon/uncordon/drain/list_nodes) affect the whole cluster, not "
        "just this namespace - say so when you use them, and only cordon/drain/uncordon when "
        "you have actual evidence (from describe_pod, events, or list_nodes) that the NODE "
        "ITSELF is unhealthy. They control scheduling eligibility only - they do not free or "
        "add any CPU/memory/GPU capacity, so they can NEVER fix a Pending pod caused by "
        "insufficient resources; do not try them for that, even as a 'let's see if this helps' "
        "step. When a pod is Pending due to insufficient CPU/memory/GPU, the only real fixes "
        "are: (1) something else in the cluster is holding that resource - use list_jobs / "
        "list_pods to check for other jobs/pods that could be deleted to free it up, and "
        "propose that to the user, or (2) the cluster genuinely doesn't have that hardware "
        "(e.g. GPU passthrough isn't set up) - say so plainly, this is outside what you can "
        "fix, and the user needs to address it at the infrastructure level. When a pod is "
        "broken, crashing, or not Running, prefer describe_pod first (it explains WHY - "
        "CrashLoopBackOff, OOMKilled, ImagePullBackOff, or an unschedulable/Pending reason "
        "such as insufficient GPU) over guessing from logs alone. If a Service seems "
        "unreachable, use check_service_endpoints to check for a selector/label mismatch "
        "before assuming anything else is wrong. Be concise."
    )
    return ChatConfig(
        system_prompt=system_prompt,
        tool_schemas=[_build_tool_schema(fn) for fn in tools.values()],
    )


# ----------------------------------------------------------------------------
# Stepped turn execution - pauses before destructive tool calls
# ----------------------------------------------------------------------------
@dataclass
class TurnResult:
    kind: Literal["final", "confirm"]
    text: str = ""
    calls: list[Any] = field(default_factory=list)                # all tool_calls in this response, in order
    executed_parts: dict[int, dict] = field(default_factory=dict)  # index -> already-executed tool-result message

    def destructive_calls(self) -> list[Any]:
        """The subset of `calls` still awaiting a user decision."""
        return [tc for i, tc in enumerate(self.calls) if i not in self.executed_parts]


def _trim_history(messages: list[dict]) -> list[dict]:
    """Caps how much conversation history gets sent to the API - every call
    resends the whole history (these APIs are stateless), so an unbounded
    conversation means unbounded, ever-growing token cost per message. Only
    trims what's SENT, not what's stored: `messages` (kept in full in
    st.session_state for the UI's scrollback) is never mutated here.

    Cuts at a "user" message boundary, never in the middle of a turn - a
    "tool" message must always immediately follow the "assistant" message
    that requested it, or the API rejects the request as malformed."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    tail = messages[-MAX_HISTORY_MESSAGES:]
    for i, msg in enumerate(tail):
        if msg.get("role") == "user":
            return tail[i:]
    return tail  # no user message in the tail (shouldn't happen) - send as-is


def _generate_with_retry(messages: list[dict], chat_config: ChatConfig, model: str):
    """Calls chat.completions.create, retrying with exponential backoff only
    on transient errors (rate-limited / server overloaded). A 404 (bad model
    name) or 401/403 (bad key) fails immediately - retrying can't fix those."""
    full_messages = [{"role": "system", "content": chat_config.system_prompt}] + _trim_history(messages)
    delay = RETRY_BASE_DELAY_S
    for attempt in range(MAX_GENERATE_RETRIES):
        try:
            return groq_client.chat.completions.create(
                model=model, messages=full_messages, tools=chat_config.tool_schemas or None
            )
        except RETRYABLE_ERRORS:
            if attempt == MAX_GENERATE_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _execute(tool_call, tools: dict[str, callable]) -> dict:
    fn = tools.get(tool_call.function.name)
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        # some models emit "null" instead of "{}" for zero-argument tools -
        # that parses fine as JSON (-> Python None) but isn't a dict, and
        # fn(**None) blows up; treat anything non-dict as "no arguments".
        args = {}
    if fn is None:
        result = f"Unknown tool: {tool_call.function.name}"
    else:
        try:
            result = fn(**args)
        except Exception as exc:
            result = f"Error: {exc}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}


def _assistant_message(msg) -> dict:
    """Reconstructs the assistant turn as a plain dict with only the fields
    the API actually needs, instead of trusting msg.model_dump() to round-trip
    cleanly through whatever extra fields the SDK's response model adds."""
    out: dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def run_turn(
    messages: list[dict],
    tools: dict[str, callable],
    chat_config: ChatConfig,
    model: str = GROQ_MODEL,
) -> TurnResult:
    """Runs the turn to completion, or until a destructive tool call needs
    confirmation. Mutates `messages` in place (appends the assistant's turn,
    and tool-result messages once a batch has none pending confirmation) so
    conversation memory persists across turns and across pause/resume."""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = _generate_with_retry(messages, chat_config, model)
        msg = response.choices[0].message
        messages.append(_assistant_message(msg))

        calls = msg.tool_calls or []
        if not calls:
            return TurnResult(kind="final", text=msg.content or "")

        if any(tc.function.name in CONFIRM_TOOLS for tc in calls):
            executed = {
                i: _execute(tc, tools) for i, tc in enumerate(calls) if tc.function.name not in CONFIRM_TOOLS
            }
            return TurnResult(kind="confirm", calls=calls, executed_parts=executed)

        for tc in calls:
            messages.append(_execute(tc, tools))

    return TurnResult(
        kind="final",
        text="Stopped after too many tool calls in a row - try rephrasing your request.",
    )


def resume_turn(
    messages: list[dict],
    tools: dict[str, callable],
    chat_config: ChatConfig,
    pending: TurnResult,
    approved: bool,
    model: str = GROQ_MODEL,
) -> TurnResult:
    """Resolves a "confirm" TurnResult - executes (or skips, if declined) the
    calls that were awaiting confirmation, in their original order relative
    to the calls that already auto-executed, then continues the loop."""
    for i, tc in enumerate(pending.calls):
        if i in pending.executed_parts:
            messages.append(pending.executed_parts[i])
        elif approved:
            messages.append(_execute(tc, tools))
        else:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "User declined this action."})
    return run_turn(messages, tools, chat_config, model)
