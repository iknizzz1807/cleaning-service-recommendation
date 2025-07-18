document.addEventListener("DOMContentLoaded", () => {
  // --- CONSTANTS ---
  const MESS_TYPES = [
    "Disorganized_pillow",
    "Messy Table",
    "dirty_bathtub",
    "dirty_floor",
    "disorganized_towel",
    "mess",
    "messy_bed",
    "messy_sink",
    "messy_table",
  ];

  const DEFAULT_MESS_ASSIGNMENTS = {
    "Messy Table": "General mess cleaning",
    mess: "General mess cleaning",
    Disorganized_pillow: "Organizing pillows",
    messy_bed: "Organizing bed",
    dirty_floor: "Surface cleaning",
    messy_sink: "Cleaning sink",
    dirty_bathtub: "Cleaning bathtub",
    disorganized_towel: "Organizing towel",
  };

  let allServices = new Set();
  let messAssignments = {}; // e.g., { "dirty_floor": ["Standard Cleaning"], "messy_bed": ["Standard Cleaning"] }
  let activeMess = null;

  // --- DOM ELEMENT SELECTORS ---
  // Uploader
  const fileInput = document.getElementById("file-input");
  const submitButton = document.getElementById("submit-btn");
  const fileNameDisplay = document.getElementById("file-name");
  const previewContainer = document.getElementById("preview-container");
  const resultContainer = document.getElementById("result-container");
  const btnText = document.querySelector(".btn-text");
  const spinner = document.querySelector(".spinner");

  // Tabs
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabPanels = document.querySelectorAll(".tab-panel");

  // Settings Panel
  const messListContainer = document.getElementById("mess-list-container");
  const newServiceInput = document.getElementById("new-service-input");
  const addServiceBtn = document.getElementById("add-service-btn");
  const serviceChecklistContainer = document.getElementById(
    "service-checklist-container"
  );

  // --- NEW: Scoring & Ranking Panel Selectors ---
  const messSeverityListContainer = document.getElementById(
    "mess-severity-list-container"
  );

  function init() {
    initializeMessAssignments();
    setupEventListeners();
    renderMessList();
    renderServiceChecklist();
    renderSeverityList(); // <-- NEW: Render the new scoring UI
  }

  function initializeMessAssignments() {
    allServices.clear();
    messAssignments = {};

    for (const mess in DEFAULT_MESS_ASSIGNMENTS) {
      const service = DEFAULT_MESS_ASSIGNMENTS[mess];
      allServices.add(service);
      if (!messAssignments[mess]) {
        messAssignments[mess] = [];
      }
      messAssignments[mess].push(service);
    }

    MESS_TYPES.forEach((mess) => {
      if (!messAssignments[mess]) {
        messAssignments[mess] = [];
      }
    });
  }

  function setupEventListeners() {
    fileInput.addEventListener("change", handleFileSelect);
    submitButton.addEventListener("click", handleUpload);
    tabButtons.forEach((button) =>
      button.addEventListener("click", () =>
        handleTabSwitch(button.dataset.tab)
      )
    );
    addServiceBtn.addEventListener("click", handleAddNewService);
    messListContainer.addEventListener("click", handleSelectMess);
    serviceChecklistContainer.addEventListener(
      "click",
      handleServiceInteraction
    );
    // --- NEW: Event listener for the severity list container ---
    messSeverityListContainer.addEventListener("input", handleSeverityChange);
  }

  function handleTabSwitch(targetTab) {
    tabPanels.forEach((panel) =>
      panel.classList.toggle("active", panel.id === `${targetTab}-panel`)
    );
    tabButtons.forEach((button) =>
      button.classList.toggle("active", button.dataset.tab === targetTab)
    );
  }

  // --- RENDER FUNCTIONS ---

  function renderMessList() {
    messListContainer.innerHTML = "";
    MESS_TYPES.forEach((mess) => {
      const messItem = document.createElement("div");
      messItem.className = "mess-item";
      messItem.textContent = mess.replace(/_/g, " ");
      messItem.dataset.mess = mess;
      if (mess === activeMess) {
        messItem.classList.add("selected");
      }
      messListContainer.appendChild(messItem);
    });
  }

  function renderServiceChecklist() {
    serviceChecklistContainer.innerHTML = "";
    serviceChecklistContainer.classList.toggle("disabled", !activeMess);

    if (allServices.size === 0) {
      serviceChecklistContainer.innerHTML = `<p class="subtle-text-color">No services created yet. Create one above.</p>`;
      return;
    }

    const sortedServices = Array.from(allServices).sort();

    sortedServices.forEach((serviceName) => {
      const isChecked =
        activeMess && messAssignments[activeMess].includes(serviceName);
      const item = document.createElement("div");
      item.className = "service-check-item";
      item.innerHTML = `
        <label>
          <input type="checkbox" data-service="${serviceName}" ${
        isChecked ? "checked" : ""
      }>
          <span>${serviceName}</span>
        </label>
        <button class="delete-btn" data-service-delete="${serviceName}" title="Delete Service">×</button>
      `;
      serviceChecklistContainer.appendChild(item);
    });
  }

  // --- NEW: Function to render the mess severity list ---
  function renderSeverityList() {
    messSeverityListContainer.innerHTML = "";
    MESS_TYPES.forEach((mess) => {
      const item = document.createElement("div");
      item.className = "severity-item";
      item.innerHTML = `
        <span class="mess-name">${mess.replace(/_/g, " ")}</span>
        <input type="range" min="0.5" max="2" value="1" step="0.1" data-mess-severity="${mess}">
        <span class="severity-value">1.0</span>
      `;
      messSeverityListContainer.appendChild(item);
    });
  }

  // --- EVENT HANDLERS ---

  function handleSelectMess(event) {
    const messItem = event.target.closest(".mess-item");
    if (messItem) {
      activeMess = messItem.dataset.mess;
      renderMessList();
      renderServiceChecklist();
    }
  }

  function handleAddNewService() {
    const serviceName = newServiceInput.value.trim();
    if (serviceName && !allServices.has(serviceName)) {
      allServices.add(serviceName);
      newServiceInput.value = "";
      renderServiceChecklist();
    }
  }

  function handleServiceInteraction(event) {
    const target = event.target;
    if (target.matches('input[type="checkbox"]')) {
      const serviceName = target.dataset.service;
      const assignedServices = messAssignments[activeMess];
      if (target.checked) {
        if (!assignedServices.includes(serviceName)) {
          assignedServices.push(serviceName);
        }
      } else {
        messAssignments[activeMess] = assignedServices.filter(
          (s) => s !== serviceName
        );
      }
    } else if (target.matches(".delete-btn")) {
      const serviceToDelete = target.dataset.serviceDelete;
      if (
        confirm(
          `Are you sure you want to delete the "${serviceToDelete}" service? It will be removed from all mess types.`
        )
      ) {
        allServices.delete(serviceToDelete);
        for (const mess in messAssignments) {
          messAssignments[mess] = messAssignments[mess].filter(
            (s) => s !== serviceToDelete
          );
        }
        renderServiceChecklist();
      }
    }
  }

  // --- NEW: Event handler for severity slider changes ---
  function handleSeverityChange(event) {
    const slider = event.target;
    if (slider.matches('input[type="range"]')) {
      const valueDisplay = slider.nextElementSibling;
      if (valueDisplay && valueDisplay.classList.contains("severity-value")) {
        valueDisplay.textContent = parseFloat(slider.value).toFixed(1);
      }
    }
  }

  // --- API INTERACTION ---
  function getServiceCentricPayload() {
    const payload = {};
    allServices.forEach((service) => {
      payload[service] = [];
    });
    for (const mess in messAssignments) {
      messAssignments[mess].forEach((service) => {
        if (payload[service]) {
          payload[service].push(mess);
        }
      });
    }
    for (const service in payload) {
      if (payload[service].length === 0) {
        delete payload[service];
      }
    }
    return payload;
  }

  // --- NEW: Function to get scoring & ranking settings (for future use) ---
  function getScoringAndRankingSettings() {
    const method = document.querySelector(
      'input[name="score-method"]:checked'
    ).value;
    const sortingRule = document.getElementById("sorting-rule-select").value;
    const severities = {};
    document.querySelectorAll("[data-mess-severity]").forEach((slider) => {
      severities[slider.dataset.messSeverity] = parseFloat(slider.value);
    });

    return {
      method,
      severities,
      sortingRule,
    };
  }

  function handleFileSelect(event) {
    resetUploaderUI();
    const file = event.target.files[0];
    if (file) {
      fileNameDisplay.textContent = file.name;
      displayPreview(file);
    }
  }

  function displayPreview(file) {
    previewContainer.innerHTML = "";
    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      previewContainer.appendChild(img);
    }
  }

  async function handleUpload() {
    const file = fileInput.files[0];
    if (!file) {
      displayError("Please select a file first.");
      return;
    }

    showLoadingState(true);
    const formData = new FormData();
    formData.append("file", file);

    const servicesPayload = getServiceCentricPayload();
    formData.append("services", JSON.stringify(servicesPayload));

    // For future implementation: you can send these settings to the backend
    const scoringSettings = getScoringAndRankingSettings();
    // formData.append("scoring_settings", JSON.stringify(scoringSettings));
    console.log("Sending payload:", servicesPayload);
    console.log(
      "Current Scoring/Ranking Settings (for demo):",
      scoringSettings
    );

    try {
      const response = await fetch("http://localhost:8000/api", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Server error: ${response.status}`);
      }
      displayResult(data);
    } catch (error) {
      console.error("Error:", error);
      displayError(
        error.message || "Upload failed. Check console for details."
      );
    } finally {
      showLoadingState(false);
    }
  }

  // --- UI STATE & DISPLAY FUNCTIONS ---

  function showLoadingState(isLoading) {
    submitButton.disabled = isLoading;
    spinner.hidden = !isLoading;
    btnText.textContent = isLoading ? "Processing..." : "Upload";
    if (isLoading) {
      resultContainer.style.display = "none";
    }
  }

  function displayResult(data) {
    resultContainer.className = "result-container success";

    let servicesHTML = "<p>No specific services required.</p>";

    if (data.services_needed && data.services_needed.length > 0) {
      const sortedServices = data.services_needed.sort(
        (a, b) => b.score - a.score
      );

      servicesHTML = `<ul>${sortedServices
        .map(
          (s) =>
            `<li>${s.name} <span style="color: #555;">(Score: ${s.score.toFixed(
              2
            )})</span></li>`
        )
        .join("")}</ul>`;
    }

    resultContainer.innerHTML = `
      <p><strong>Processing Complete!</strong></p>
      <p><strong>Suggested Services (sorted by relevance):</strong></p>
      ${servicesHTML}
      <p><strong>Result Image:</strong></p>
      <img src="${data.result_image}" alt="Processed Image" style="max-width: 100%; border-radius: 8px; margin-top: 10px;" />
    `;

    resultContainer.style.display = "block";
    previewContainer.innerHTML = "";
  }

  function displayError(message) {
    resultContainer.className = "result-container error";
    resultContainer.innerHTML = `<p><strong>Error:</strong> ${message}</p>`;
    resultContainer.style.display = "block";
  }

  function resetUploaderUI() {
    previewContainer.innerHTML = "";
    resultContainer.innerHTML = "";
    resultContainer.style.display = "none";
    fileNameDisplay.textContent = "No file chosen";
  }

  init();
});
