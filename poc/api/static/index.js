(() => {
  "use strict";

  const backend = document.getElementById("agent-backend");
  const provider = document.getElementById("agent-provider");
  const model = document.getElementById("agent-model");
  const executionMode = document.getElementById("execution-mode");
  const swarmStrategy = document.getElementById("swarm-strategy");
  const fields = document.querySelectorAll("[data-model-field]");

  const syncModelFields = () => {
    const usesPydanticAI = backend.value === "pydantic_ai";
    const usesExternalModel = backend.value !== "custom_python";
    fields.forEach((field) => field.classList.toggle("is-disabled", !usesExternalModel));
    provider.disabled = !usesExternalModel;
    model.disabled = !usesExternalModel;
    model.required = usesPydanticAI;
  };

  const syncSwarmStrategy = () => {
    const hybrid = swarmStrategy.querySelector('option[value="hybrid_v1"]');
    const boardMode = executionMode.value === "board_claim";
    hybrid.disabled = !boardMode;
    if (!boardMode && swarmStrategy.value === "hybrid_v1") swarmStrategy.value = "board";
  };

  backend.addEventListener("change", syncModelFields);
  executionMode.addEventListener("change", syncSwarmStrategy);
  syncModelFields();
  syncSwarmStrategy();
})();
