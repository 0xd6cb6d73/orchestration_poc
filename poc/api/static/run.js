(() => {
  "use strict";

  const runId = document.body.dataset.runId;
  const initialNode = document.getElementById("initial-status");
  const connection = document.getElementById("connection-state");
  let timer = null;
  let refreshInFlight = false;

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const empty = (message) => element("div", "empty-inline", message);

  const badge = (status) => {
    const node = element("span", "status-badge", String(status).replaceAll("_", " "));
    node.dataset.status = status;
    return node;
  };

  const modelLabel = (runtime) => {
    if (!runtime.model) {
      return runtime.backend === "custom_python" ? "built-in deterministic" : "role default";
    }
    if (!runtime.provider || runtime.model.includes(":")) return runtime.model;
    return `${runtime.provider}:${runtime.model}`;
  };

  const render = (status) => {
    const run = status.run;
    document.title = `${run.status} · ${runId} · Agent Operations`;
    const runStatus = document.getElementById("run-status");
    runStatus.textContent = String(run.status).replaceAll("_", " ");
    runStatus.dataset.status = run.status;
    document.getElementById("run-objective").textContent = status.plan?.objective || run.objective;
    document.getElementById("plan-version").textContent = status.plan?.version || "—";
    const runtime = status.plan?.agent_runtime || {};
    document.getElementById("agent-runtime").textContent = runtime.backend || "custom_python";
    document.getElementById("execution-mode").textContent = String(
      status.plan?.execution_mode || "hierarchical_dag",
    ).replaceAll("_", " ");
    document.getElementById("swarm-strategy").textContent = String(
      status.plan?.swarm_strategy || "board",
    ).replaceAll("_", " ");
    document.getElementById("agent-model").textContent = modelLabel(runtime);
    document.getElementById("agent-count").textContent = status.agents.length;
    document.getElementById("workflow-count").textContent = status.workflows.length;
    document.getElementById("artifact-count").textContent = status.artifacts.length;
    document.getElementById("event-count").textContent = status.events.length;

    const cancelForm = document.querySelector("[data-cancel-form]");
    if (cancelForm) cancelForm.hidden = run.status !== "running";

    renderAgents(status.agents);
    renderHybrid(status.hybrid || {}, status.plan);
    renderWorkflows(status.workflows);
    renderPlan(status.plan);
    renderArtifacts(status.artifacts);
    renderEvents(status.events, status.agents);

    if (["completed", "failed", "cancelled"].includes(run.status)) {
      if (timer) window.clearInterval(timer);
      timer = null;
      connection.innerHTML = "<span class=\"live-dot\"></span> settled";
    }
  };

  const renderHybrid = (hybrid, plan) => {
    const panel = document.getElementById("hybrid-panel");
    const enabled = plan?.swarm_strategy === "hybrid_v1";
    panel.hidden = !enabled;
    if (!enabled) return;

    const rounds = hybrid.collaboration_rounds || [];
    const phase = rounds.length ? rounds[rounds.length - 1].phase : "framed";
    const phaseNode = document.getElementById("hybrid-phase");
    phaseNode.textContent = phase.replaceAll("_", " ");
    phaseNode.dataset.status = phase === "complete" ? "completed" : "running";

    const candidates = document.getElementById("candidate-list");
    candidates.replaceChildren();
    (hybrid.candidates || []).forEach((candidate) => {
      const row = element("article", "candidate-row");
      const heading = element("div", "candidate-heading");
      heading.append(element("strong", "", candidate.hypothesis_key.replaceAll("_", " ")));
      heading.append(badge(candidate.state));
      row.append(heading);
      const evaluation = candidate.evaluation;
      row.append(
        element(
          "span",
          "candidate-detail",
          evaluation
            ? `validity ${evaluation.validity}/5 · evidence ${evaluation.evidence}/5 · novelty ${evaluation.novelty}/5`
            : candidate.visibility.replaceAll("_", " "),
        ),
      );
      const link = element("a", "candidate-artifact", candidate.artifact_id);
      link.href = `/artifacts/${encodeURIComponent(candidate.artifact_id)}`;
      row.append(link);
      candidates.append(row);
    });
    if (!candidates.children.length) candidates.append(empty("Waiting for sealed proposals."));

    const assurance = document.getElementById("assurance-list");
    assurance.replaceChildren();
    [...(hybrid.verifications || []), ...(hybrid.delivery_decisions || [])].forEach((item) => {
      const row = element("article", "assurance-row");
      const state = item.verdict || (item.permitted ? "permitted" : "blocked");
      row.append(badge(state));
      row.append(
        element(
          "span",
          "",
          item.verification_id || item.delivery_decision_id || "acceptance decision",
        ),
      );
      assurance.append(row);
    });
    if (!assurance.children.length) assurance.append(empty("Independent checks are pending."));

    const blackboard = document.getElementById("blackboard-list");
    blackboard.replaceChildren();
    (hybrid.blackboard || []).slice().reverse().forEach((record) => {
      const row = element("article", "blackboard-row");
      row.append(badge(record.epistemic_status));
      row.append(element("span", "", record.concise_statement));
      blackboard.append(row);
    });
    if (!blackboard.children.length) blackboard.append(empty("No evidence claims published."));
  };

  const renderAgents = (agents) => {
    const target = document.getElementById("agent-tree");
    target.replaceChildren();
    if (!agents.length) {
      target.append(empty("Agents appear after the plan is approved."));
      return;
    }

    const byId = new Map(agents.map((agent) => [agent.agent_instance_id, agent]));
    const children = new Map();
    agents.forEach((agent) => {
      const parent = agent.parent_agent_id;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(agent);
    });
    children.forEach((items) => items.sort((a, b) => a.role_id.localeCompare(b.role_id)));
    const roots = agents.filter(
      (agent) => !agent.parent_agent_id || !byId.has(agent.parent_agent_id),
    );
    const rootList = element("ul");
    roots.forEach((agent) => rootList.append(renderAgentNode(agent, children)));
    target.append(rootList);
  };

  const renderAgentNode = (agent, children) => {
    const item = element("li");
    const card = element("article", "agent-card");
    card.dataset.tier = agent.tier;
    card.title = agent.agent_instance_id;

    const head = element("div", "agent-card-head");
    head.append(element("span", "agent-role", agent.role_id.replaceAll("_", " ")));
    const state = element("i", "agent-state");
    if (agent.status !== "active") state.classList.add("inactive");
    state.title = agent.status;
    head.append(state);
    card.append(head);
    card.append(element("code", "agent-id", agent.agent_instance_id));

    const meta = element("div", "agent-meta");
    meta.append(element("span", "", agent.tier === "sub_orchestrator" ? "supervisor" : agent.tier));
    meta.append(element("span", "", agent.agent_backend || "custom_python"));
    if (agent.agent_model) {
      meta.append(
        element(
          "span",
          "",
          modelLabel({provider: agent.agent_provider, model: agent.agent_model}),
        ),
      );
    }
    card.append(meta);
    item.append(card);

    const descendants = children.get(agent.agent_instance_id) || [];
    if (descendants.length) {
      const list = element("ul");
      descendants.forEach((child) => list.append(renderAgentNode(child, children)));
      item.append(list);
    }
    return item;
  };

  const renderWorkflows = (workflows) => {
    const target = document.getElementById("workflow-list");
    target.replaceChildren();
    if (!workflows.length) {
      target.append(empty("No workflows have been dispatched yet."));
      return;
    }
    workflows.forEach((workflow) => {
      const card = element("article", "workflow-card");
      const head = element("div", "workflow-head");
      head.append(element("span", "workflow-name", workflow.workflow_id));
      head.append(badge(workflow.status));
      card.append(head);

      const tasks = workflow.tasks || [];
      const finished = tasks.filter((task) =>
        ["succeeded", "failed", "cancelled"].includes(task.status),
      ).length;
      const track = element("div", "progress-track");
      const fill = element("div", "progress-fill");
      fill.style.width = `${tasks.length ? (finished / tasks.length) * 100 : 0}%`;
      track.append(fill);
      card.append(track);

      const taskList = element("div", "task-list");
      tasks.forEach((task) => {
        const row = element("div", "task-row");
        row.append(element("i", `task-dot ${task.status}`));
        row.append(element("span", "", task.task_id.replaceAll("_", " ")));
        row.append(
          element("code", "task-worker", task.agent_instance_id || task.status.replaceAll("_", " ")),
        );
        taskList.append(row);
      });
      card.append(taskList);
      target.append(card);
    });
  };

  const renderPlan = (plan) => {
    const target = document.getElementById("plan-detail");
    target.replaceChildren();
    if (!plan) {
      target.append(empty("Plan unavailable."));
      return;
    }
    const runtime = plan.agent_runtime || {};
    target.append(
      planList("Execution", [
        `runtime: ${runtime.backend || "custom_python"}`,
        `orchestration: ${String(plan.execution_mode || "hierarchical_dag").replaceAll("_", " ")}`,
        `swarm strategy: ${String(plan.swarm_strategy || "board").replaceAll("_", " ")}`,
        `model: ${modelLabel(runtime)}`,
      ]),
    );
    target.append(planList("Constraints", plan.constraints));
    target.append(planTokens("Permitted tools", plan.permitted_tools));
    target.append(planList("Completion criteria", plan.completion_criteria));
  };

  const planList = (title, values) => {
    const section = element("section", "plan-section");
    section.append(element("h3", "", title));
    const list = element("ul");
    (values || []).forEach((value) => list.append(element("li", "", value)));
    section.append(list);
    return section;
  };

  const planTokens = (title, values) => {
    const section = element("section", "plan-section");
    section.append(element("h3", "", title));
    const list = element("div", "token-list");
    (values || []).forEach((value) => list.append(element("span", "", value)));
    section.append(list);
    return section;
  };

  const renderArtifacts = (artifacts) => {
    const target = document.getElementById("artifact-list");
    target.replaceChildren();
    if (!artifacts.length) {
      target.append(empty("Published agent outputs will appear here."));
      return;
    }
    artifacts.slice().reverse().forEach((artifact) => {
      const row = element("a", "artifact-row");
      row.href = `/artifacts/${encodeURIComponent(artifact.artifact_id)}`;
      row.append(element("span", "artifact-icon", "◇"));
      const copy = element("span");
      copy.append(element("span", "artifact-title", artifact.artifact_id));
      copy.append(
        element(
          "span",
          "artifact-detail",
          `${artifact.media_type} · ${artifact.producer_task_id || "runtime"}`,
        ),
      );
      row.append(copy);
      row.append(element("span", "artifact-arrow", "↗"));
      target.append(row);
    });
  };

  const renderEvents = (events, agents) => {
    const target = document.getElementById("event-list");
    target.replaceChildren();
    if (!events.length) {
      target.append(empty("The durable trajectory is empty."));
      return;
    }
    const roles = new Map(agents.map((agent) => [agent.agent_instance_id, agent.role_id]));
    events.slice(-40).reverse().forEach((event) => {
      const family = event.event_type.split(".")[0];
      const row = element("article", `event-row is-${family}`);
      row.append(element("span", "event-name", event.event_type.replaceAll("_", " ")));
      row.append(element("span", "event-seq", `#${event.seq}`));
      const actor = roles.get(event.actor_id) || event.actor_id || "runtime";
      const details = eventDetails(event.data);
      row.append(element("span", "event-detail", details ? `${actor} · ${details}` : actor));
      target.append(row);
    });
  };

  const eventDetails = (data) => {
    if (!data) return "";
    const fields = ["task_id", "child_role", "tool", "outcome", "workflow_id", "artifact_id"];
    return fields
      .filter((field) => data[field] !== undefined && data[field] !== null)
      .slice(0, 2)
      .map((field) => String(data[field]))
      .join(" · ");
  };

  const refresh = async () => {
    if (refreshInFlight) return;
    refreshInFlight = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch(`/runs/${encodeURIComponent(runId)}`, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`status ${response.status}`);
      connection.innerHTML = "<span class=\"live-dot\"></span> live";
      render(await response.json());
      connection.classList.remove("is-stale");
    } catch (_error) {
      connection.classList.add("is-stale");
      connection.innerHTML = "<span class=\"live-dot\"></span> reconnecting";
    } finally {
      window.clearTimeout(timeout);
      refreshInFlight = false;
    }
  };

  document.getElementById("refresh-run").addEventListener("click", refresh);
  document.getElementById("copy-run-id").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.querySelector("span").textContent = "copied";
    } catch (_error) {
      button.querySelector("span").textContent = "select ID";
    }
  });

  try {
    render(JSON.parse(initialNode.textContent));
    const initialStatus = JSON.parse(initialNode.textContent).run.status;
    if (!["completed", "failed", "cancelled"].includes(initialStatus)) {
      timer = window.setInterval(refresh, 1800);
    }
  } catch (_error) {
    refresh();
  }
})();
