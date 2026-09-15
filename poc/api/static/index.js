(() => {
  "use strict";

  const backend = document.getElementById("agent-backend");
  const provider = document.getElementById("agent-provider");
  const model = document.getElementById("agent-model");
  const executionMode = document.getElementById("execution-mode");
  const swarmStrategy = document.getElementById("swarm-strategy");
  const fields = document.querySelectorAll("[data-model-field]");

  const syncModelFields = () => {
    const usesPydanticAI = ["pydantic_ai", "semantic_pydantic_ai"].includes(backend.value);
    const usesExternalModel = backend.value !== "custom_python";
    fields.forEach((field) => field.classList.toggle("is-disabled", !usesExternalModel));
    provider.disabled = !usesExternalModel;
    model.disabled = !usesExternalModel;
    model.required = usesPydanticAI;
  };

  const syncSwarmStrategy = () => {
    const boardMode = executionMode.value === "board_claim";
    ["hybrid_v1", "hybrid_v2"].forEach((value) => {
      const option = swarmStrategy.querySelector(`option[value="${value}"]`);
      option.disabled = !boardMode;
    });
    if (!boardMode && ["hybrid_v1", "hybrid_v2"].includes(swarmStrategy.value)) {
      swarmStrategy.value = "board";
    }
  };

  backend.addEventListener("change", syncModelFields);
  executionMode.addEventListener("change", syncSwarmStrategy);
  syncModelFields();
  syncSwarmStrategy();
})();
