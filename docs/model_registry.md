# Model Registry

Use this page to search and download trained RouteE Powertrain models.

<!-- Styling for the dynamic portions of the page. -->
<style>
  .dashboard-container {
    max-width: 100%;
    margin: 20px 0;
    color: var(--pst-color-text-base, #333);
  }
  .filters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--pst-color-surface, #f8f9fa);
    border-radius: 8px;
    border: 1px solid var(--pst-color-border, #ddd);
  }
  .filters-grid input, .filters-grid select {
    padding: 8px 12px;
    border: 1px solid var(--pst-color-border, #ccc);
    border-radius: 6px;
    font-size: 0.95rem;
    background-color: var(--pst-color-background, #fff);
    color: var(--pst-color-text-base, #333);
  }
  .filters-grid input:focus, 
  .filters-grid select:focus,
  .version-select:focus {
    border-color: #ff7f0e;
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 127, 14, 0.25);
  }
  #model-search-input {
    grid-column: 1 / -1;
  }
  .result-card {
    border: 1px solid var(--pst-color-border, #ddd);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    background: var(--pst-color-background, #fff);
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }
  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 10px;
  }
  .result-title {
    margin: 0;
    font-size: 1.3rem;
    color: #0f6cbd;
    font-weight: 600;
  }
  .feature-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    list-style: none;
    padding: 0;
    margin: 8px 0 16px 0;
  }
  .feature-list li {
    background: rgba(15, 108, 189, 0.1);
    color: #0f6cbd;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
  }
  .snippet-wrapper {
    position: relative;
    margin: 12px 0;
  }
  .snippet-box {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 12px 40px 12px 12px;
    border-radius: 6px;
    font-family: var(--pst-font-family-monospace, monospace);
    font-size: 0.85rem;
    white-space: pre-wrap;
    overflow-x: auto;
  }
  .copy-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(255, 255, 255, 0.15);
    border: none;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .copy-btn:hover {
    background: rgba(255, 255, 255, 0.3);
  }
  .actions-row {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 16px;
  }
  .feature-set-card {
    border: 1px solid var(--pst-color-border, #e5e5e5);
    border-radius: 8px;
    padding: 14px;
    margin-top: 12px;
    background: var(--pst-color-surface, #fafafa);
  }
  .feature-set-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    color: var(--pst-color-text-base, #333);
  }
  .version-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--pst-color-text-muted, #777);
  }
  .btn-download {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    background: #0f6cbd;
    color: white !important;
    text-decoration: none !important;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
  }
  .btn-download:hover {
    background: #0a4b85;
  }
  .version-select-wrapper {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .version-select {
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid var(--pst-color-border, #ccc);
    background-color: var(--pst-color-background, #fff);
    color: var(--pst-color-text-base, #333);
    font-size: 0.85rem;
  }
  #status-message {
    font-size: 1.1rem;
    text-align: center;
    padding: 40px 20px;
    color: var(--pst-color-text-muted, #666);
  }

  .snippet-box .token-keyword { color: #c586c0; }
  .snippet-box .token-variable { color: #9cdcfe; }
  .snippet-box .token-function { color: #dcdcaa; }
  .snippet-box .token-string { color: #ce9178; }
  .snippet-box .token-punctuation { color: #d4d4d4; }
</style>

<!-- Base div that everything gets dynamically added into -->
<div class="dashboard-container">
  <div class="filters-grid">
    <input type="text" id="model-search-input" placeholder="Search models, powertrain, features...">
    <select id="make-filter"><option value="">All Makes</option></select>
    <select id="model-filter"><option value="">All Models</option></select>
    <select id="powertrain-filter"><option value="">All Powertrains</option></select>
    <select id="architecture-filter"><option value="">All Architectures</option></select>
    <input type="number" id="year-min" placeholder="Year Min">
    <input type="number" id="year-max" placeholder="Year Max">
  </div>

  <div id="results-container">
    <div id="status-message">Loading model registry...</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>

<!-- The script controlling the dynamic portions of the page. -->
<script>
document.addEventListener('DOMContentLoaded', () => {

  // URLs used to fetch content from Hugging Face. If repo moved, only BASE_URL needs updated.
  const BASE_URL = `https://huggingface.co/nreinicke/routee-powertrain-model-library/raw/main/`;
  const INDEX_URL = BASE_URL + `v2/index.json`;
  const HF_RESOLVE = BASE_URL.replace('/raw/main/', '/resolve/main/');
  const HF_API_BASE = BASE_URL.replace('https://huggingface.co/', 'https://huggingface.co/api/models/').replace('/raw/main/', '/tree/main/');
  let allModels = [];
  let vehicleGroups = [];
  let currentRenderedGroups = [];

  // --- DOM Elements ---
  const searchInput = document.getElementById('model-search-input');
  const makeFilter = document.getElementById('make-filter');
  const modelFilter = document.getElementById('model-filter');
  const powertrainFilter = document.getElementById('powertrain-filter');
  const yearMinInput = document.getElementById('year-min');
  const yearMaxInput = document.getElementById('year-max');
  const resultsContainer = document.getElementById('results-container');
  const architectureFilter = document.getElementById('architecture-filter');

  // --- Utility Functions ---

  /** Capitalizes words in a string. Words 3 letters or less all letters are capitalized. */
  const formatTitle = (value) => {
    if (!value) return 'Unknown';
    return String(value)
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.length <= 3 ? word.toUpperCase() : word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  /** Applies syntax highlighting to a Python snippet. */
  /** This one could be removed but it was bugging me to have code without syntax highlighting */
  const highlightPySnippet = (code) => {
    if (!code) return '';

    // Regex patterns for Python tokens
    const tokenRegex = /(".*?")|\b(import|as)\b|\b(load_model)\b|\b(pt|model)\b|([=().,])/g;

    //Wrap tokens in styling
    return code.replace(tokenRegex, (match, string, keyword, fn, variable, punctuation) => {
      if (string) {
        return `<span class="token-string">${string}</span>`;
      } else if (keyword) {
        return `<span class="token-keyword">${keyword}</span>`;
      } else if (fn) {
        return `<span class="token-function">${fn}</span>`;
      } else if (variable) {
        return `<span class="token-variable">${variable}</span>`;
      } else if (punctuation) {
        return `<span class="token-punctuation">${punctuation}</span>`;
      }
      return match;
    });
  };

  /** Formatting for architecture keys. */
  const formatArchitectureTag = (value) => {
    const labels = { random_forest: 'Random Forest', ngboost: 'NGBoost', cnn: 'CNN' };
    return labels[String(value || '').toLowerCase()] || formatTitle(value);
  };
  
  /** Parses various year formats (number, string, array) into a range and display format. */
  const parseYears = (yearData) => {
    if (!yearData) return { min: null, max: null, display: 'N/A' };
    
    if (Array.isArray(yearData)) {
      const years = yearData.map(Number).filter(n => !isNaN(n));
      if (years.length > 0) {
        const min = Math.min(...years);
        const max = Math.max(...years);
        const display = min === max ? String(min) : `${min}-${max}`;
        return { min, max, display };
      }
      return { min: null, max: null, display: 'N/A' };
    }
    
    if (typeof yearData === 'number') {
      return { min: yearData, max: yearData, display: String(yearData) };
    }
    
    return { min: null, max: null, display: 'N/A' };
  };


  // --- Data Processing ---

  /** Takes a raw model object pulls out the needed properties. */
  const processModelData = (model) => {
    const id = model.model_id || {};
    const modelIdString = String(model.path).replace('v2/', '');

    return {
      ...model,
      make: id.make || 'Unknown',
      vehicleModel: model.vehicle_model || 'Unknown',
      powertrain: model.powertrain_type || 'Unknown',
      version: parseInt(id.version, 10) || 0,
      architectureTag: model.architecture_tag || 'unknown',
      variantName: formatTitle(id.config_slug || 'Feature Set').replace("RF ", "Random Forest ").replace("NGB ", "NGBoost "),
      modelPath: model.path || '',
      yearRange: parseYears(id.year),
      pySnippet: `import routee.powertrain as pt\n\nmodel = pt.load_model("${modelIdString}")`
    };
  };

  /** Groups the flat list of models by vehicle, year, then by version tag. */
  const groupModelsByVehicle = (models) => {
    const groups = models.reduce((acc, model) => {
      const yearKey = model.yearRange ? model.yearRange.display : 'unknown';
      const key = `${model.make}|${model.vehicleModel}|${model.powertrain}|${yearKey}`.toLowerCase();
      
      if (!acc[key]) {
        acc[key] = { key, make: model.make, vehicle: model.vehicleModel, powertrain: model.powertrain, versions: {} };
      }
      if (!acc[key].versions[model.version]) {
        acc[key].versions[model.version] = [];
      }
      acc[key].versions[model.version].push(model);
      return acc;
    }, {});

    return Object.values(groups).map(group => {
      group.versions = Object.entries(group.versions)
        .map(([versionNum, featureSets]) => ({
          version: Number(versionNum),
          featureSets,
          yearRange: featureSets[0].yearRange
        }))
        .sort((a, b) => b.version - a.version);
      return group;
    });
  };

  /** Re-populates the model dropdown based on the currently selected make */
  const updateModelOptions = (models) => {
    const selectedMake = makeFilter.value;
    const currentSelectedModel = modelFilter.value;

    modelFilter.innerHTML = '<option value="">All Models</option>';
    const filteredModels = selectedMake? models.filter(m => m.make === selectedMake): models;
    const uniqueModels = [...new Set(filteredModels.map(m => m.vehicleModel).filter(Boolean))].sort();
    
    uniqueModels.forEach(m => {
      modelFilter.add(new Option(formatTitle(m), m));
    });

    if (uniqueModels.includes(currentSelectedModel)) {
      modelFilter.value = currentSelectedModel;
    } else {
      modelFilter.value = "";
    }
  };

  /** Populates the initial, static filters (Make, Powertrain, Architecture) */
  const populateFilters = (models) => {
    const getOptions = (key) => [...new Set(models.map(m => m[key]).filter(Boolean))].sort();
    
    getOptions('make').forEach(v => makeFilter.add(new Option(formatTitle(v), v)));
    getOptions('powertrain').forEach(v => powertrainFilter.add(new Option(String(v).replace(/_/g, ' '), v)));
    getOptions('architectureTag').forEach(arch => architectureFilter.add(new Option(formatArchitectureTag(arch), arch)));

    updateModelOptions(models);
  };


  // --- Rendering Functions ---

  const renderFeatureSetCard = (model) => `
    <div class="feature-set-card">
      <h3 class="feature-set-title">${model.variantName}</h3>
      <div style="font-size: 0.85rem; font-weight: bold;">Expected Features:</div>
      <ul class="feature-list">
        ${(model.feature_names || []).map(f => `<li>${f}</li>`).join('') || '<li>No features listed</li>'}
      </ul>
      <div class="snippet-wrapper">
        <button class="copy-btn">Copy</button>
        <div class="snippet-box">${highlightPySnippet(model.pySnippet)}</div>
      </div>
      <button class="btn-download" data-model-path="${model.modelPath}">Download Model</button>
    </div>`;

const renderArchitectureGroup = (arch, models) => `
    <div style="margin-top: 24px;">
      <h2 style="margin: 0 0 8px 0; font-size: 1.4rem; color: var(--pst-color-text-base, #333); font-weight: 600;">
        ${formatArchitectureTag(arch)}
      </h2>
      <hr style="border: 0; border-top: 2px solid var(--pst-color-border, #eee); margin-bottom: 16px;">
      <div style="display: grid; gap: 12px;">
        ${models.map(renderFeatureSetCard).join('')}
      </div>
    </div>`;

  const renderVersionContent = (version, filterArch = '', group = null) => {
    const query = searchInput.value.toLowerCase().trim();
    const searchTerms = query.split(/\s+/).filter(Boolean);

    const byArch = version.featureSets.reduce((acc, model) => {
      if (filterArch && model.architectureTag != filterArch) return acc;

      if (searchTerms.length > 0 && group) {
        const allYears = [];
        if (version.yearRange) {
          for (let y = version.yearRange.min; y <= version.yearRange.max; y++) {
            allYears.push(String(y));
          }
        }
        
        const searchableMetadata = [
          group.make,
          group.vehicle,
          group.powertrain,
          model.architectureTag,
          model.variantName,
          ...(model.feature_names || []),
          ...allYears
        ].map(str => String(str || '').toLowerCase());
        
        const matchesAllTerms = searchTerms.every(term => 
          searchableMetadata.some(meta => meta.includes(term))
        );
        
        if (!matchesAllTerms) return acc;
      }

      (acc[model.architectureTag] = acc[model.architectureTag] || []).push(model);
      return acc;
    }, {});

    const content = Object.entries(byArch).map(([arch, models]) => renderArchitectureGroup(arch, models)).join('');
    
    return content || '<div style="padding: 20px; text-align: center; color: #777;">No models in this version match the specific search terms.</div>';
  };

  const renderVersionSelector = (group, groupIndex) => {
    if (group.versions.length <= 1) return '';
    const options = group.versions.map((v, i) =>
      `<option value="${i}">Version ${v.version}${i === 0 ? ' (Latest)' : ''}</option>`
    ).join('');
    return `
      <div class="version-select-wrapper">
        <span>Previous Versions:</span>
        <select class="version-select" data-group-index="${groupIndex}">${options}</select>
      </div>`;
  };
  
  const getVersionYearDisplay = (range) => {
    if (!range || !range.display) return 'N/A';
    return range.display;
  };

  const renderVehicleCard = (group, groupIndex) => {
    const latestVersion = group.versions[0];
    const card = document.createElement('div');
    card.className = 'result-card';
    card.dataset.groupIndex = groupIndex;
    
    const activeArch = architectureFilter.value;

    card.innerHTML = `
      <div class="result-header">
        <div>
          <h2 class="result-title"><span class="year-display">${getVersionYearDisplay(latestVersion.yearRange)}</span> ${formatTitle(group.make)} ${formatTitle(group.vehicle)}</h2>
          <div style="font-weight: bold; text-transform: uppercase;">${String(group.powertrain).replace(/_/g, ' ')}</div>
        </div>
      </div>
      <div class="feature-sets-container">${renderVersionContent(latestVersion, activeArch, group)}</div>
      <div class="actions-row">${renderVersionSelector(group, groupIndex)}</div>
    `;
    resultsContainer.appendChild(card);
  };
  
  const renderResults = (filteredGroups) => {
    resultsContainer.innerHTML = '';
    currentRenderedGroups = filteredGroups;
    if (filteredGroups.length === 0) {
      resultsContainer.innerHTML = '<div id="status-message">No models match your search criteria.</div>';
      return;
    }
    filteredGroups.forEach(renderVehicleCard);
  };

  // --- Event Handlers ---

  const handleSearch = () => {
    // Dropdown and Year Range Filters
    const query = searchInput.value.toLowerCase().trim();
    const make = makeFilter.value;
    const model = modelFilter.value;
    const pt = powertrainFilter.value;
    const arch = architectureFilter.value;
    const yearMin = parseInt(yearMinInput.value, 10) || null;
    const yearMax = parseInt(yearMaxInput.value, 10) || null;

    const filtered = vehicleGroups.filter(group => {
      if (make && group.make !== make) return false;
      if (model && group.vehicle !== model) return false;
      if (pt && group.powertrain !== pt) return false;
      
      const latestYear = group.versions[0].yearRange;
      if (yearMin && latestYear && latestYear.max < yearMin) return false;
      if (yearMax && latestYear && latestYear.min > yearMax) return false;

      if (arch) {
        const hasMatchingArch = group.versions.some(v => v.featureSets.some(m => m.architectureTag === arch));
        if (!hasMatchingArch) return false;
      }

      // Text Search
      if (query) {
        const searchTerms = query.split(/\s+/).filter(Boolean);
        const allYears = group.versions.flatMap(v => {
          if (!v.yearRange) return [];
          const years = [];
          for (let y = v.yearRange.min; y <= v.yearRange.max; y++) {
            years.push(String(y));
          }
          return years;
        });

        const groupArchs = group.versions.flatMap(v => v.featureSets.map(m => m.architectureTag || ''));
        const groupFeatures = group.versions.flatMap(v => v.featureSets.flatMap(m => m.feature_names || []));
        const searchableMetadata = [
          group.make,
          group.vehicle,
          group.powertrain,
          ...allYears,
          ...groupArchs,
          ...groupFeatures
        ].map(str => String(str).toLowerCase());

        const matchesAllTerms = searchTerms.every(term => 
          searchableMetadata.some(meta => meta.includes(term))
        );

        if (!matchesAllTerms) return false;
      }
      
      return true;
    });
    renderResults(filtered);
  };
  
  const handleEventDelegation = (e) => {
    if (e.target.matches('.version-select')) {
      const groupIndex = e.target.dataset.groupIndex;
      const versionIndex = e.target.value;
      const group = currentRenderedGroups[groupIndex];
      const card = e.target.closest('.result-card');
      if (group && card) {
        const version = group.versions[versionIndex];
        const activeArch = architectureFilter.value;
        
        card.querySelector('.year-display').textContent = getVersionYearDisplay(version.yearRange);
        card.querySelector('.feature-sets-container').innerHTML = renderVersionContent(version, activeArch, group);
      }
    }

    if (e.target.matches('.copy-btn')) {
      const snippet = e.target.nextElementSibling.innerText;
      navigator.clipboard.writeText(snippet).then(() => {
        e.target.textContent = 'Copied!';
        setTimeout(() => { e.target.textContent = 'Copy'; }, 2000);
      });
    }

    if (e.target.matches('.btn-download')) {
      const btn = e.target;
      const modelPath = btn.dataset.modelPath;
      if (modelPath) downloadModelZip(btn, modelPath);
    }
  };

  /** Handles downloading the models - fetches all files in the model's folder and packages them into a .zip */
  const downloadModelZip = async (btn, modelPath) => {
    const HF_API = HF_API_BASE + modelPath;

    btn.textContent = 'Fetching...';
    btn.disabled = true;

    try {
      const fileList = await fetch(HF_API).then(r => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      });

      const files = fileList.filter(f => f.type === 'file');
      if (files.length === 0) throw new Error('No files found at path.');

      const zip = new JSZip();
      let completed = 0;

      await Promise.all(files.map(async (f) => {
        const response = await fetch(HF_RESOLVE + f.path);
        if (!response.ok) throw new Error(`Failed to fetch ${f.path}`);
        const blob = await response.blob();
        zip.file(f.path.split('/').pop(), blob);
        completed++;
        btn.textContent = `Downloading ${completed}/${files.length}...`;
      }));

      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(zipBlob);
      a.download = `${modelPath.replace(/\//g, '_').replace('v2_', '')}.zip`;
      a.click();
      URL.revokeObjectURL(a.href);

    } catch (err) {
      console.error('Download failed:', err);
      btn.textContent = 'Error — Retry';
      btn.disabled = false;
      return;
    }

    btn.textContent = 'Download Model';
    btn.disabled = false;
  };

  const init = async () => {
    try {
      const response = await fetch(INDEX_URL);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const rawData = await response.json();
      
      allModels = (rawData.models || []).map(processModelData);
      vehicleGroups = groupModelsByVehicle(allModels);
      
      populateFilters(allModels);
      renderResults(vehicleGroups);

    } catch (error) {
      resultsContainer.innerHTML = `<div id="status-message" style="color: #d9534f;"><strong>Error: Failed to load model catalog.</strong><br><span>Could not fetch data from ${INDEX_URL}.</span></div>`;
      console.error('Failed to load index.json:', error);
    }
  };

  [searchInput, makeFilter, modelFilter, powertrainFilter, architectureFilter, yearMinInput, yearMaxInput].forEach(el => 
    el.addEventListener('input', handleSearch)
  );

  makeFilter.addEventListener('change', () => {
    updateModelOptions(allModels);
    handleSearch();
  });

  resultsContainer.addEventListener('change', handleEventDelegation);
  resultsContainer.addEventListener('click', handleEventDelegation);

  init();
});
</script>
