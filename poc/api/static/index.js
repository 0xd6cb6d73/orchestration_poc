(() => {
  "use strict";

  const backend = document.getElementById("agent-backend");
  const provider = document.getElementById("agent-provider");
  const model = document.getElementById("agent-model");
  const fields = document.querySelectorAll("[data-model-field]");

  const syncModelFields = () => {
    const usesPydanticAI = backend.value === "pydantic_ai";
    const usesExternalModel = backend.value !== "custom_python";
    fields.forEach((field) => field.classList.toggle("is-disabled", !usesExternalModel));
    provider.disabled = !usesExternalModel;
    model.disabled = !usesExternalModel;
    model.required = usesPydanticAI;
  };

  backend.addEventListener("change", syncModelFields);
  syncModelFields();
})();
