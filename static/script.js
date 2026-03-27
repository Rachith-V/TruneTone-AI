// ==============================
// FILE UPLOAD HANDLING
// ==============================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const resultContent = document.getElementById('resultContent');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Single file upload
selectFileBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        predictBtn.disabled = false;
        uploadArea.style.borderColor = 'var(--success-color)';
        uploadArea.innerHTML = `
            <p>✅ File selected: ${selectedFile.name}</p>
            <p style="color: var(--success-color); margin-top: 10px;">Ready for analysis</p>
        `;
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        fileInput.files = files;
        predictBtn.disabled = false;
        uploadArea.style.borderColor = 'var(--success-color)';
        uploadArea.innerHTML = `
            <p>✅ File selected: ${selectedFile.name}</p>
            <p style="color: var(--success-color); margin-top: 10px;">Ready for analysis</p>
        `;
    }
});

// ==============================
// PREDICTION
// ==============================

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loading.style.display = 'block';
    predictBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data.result);
            resultsSection.style.display = 'block';
        } else {
            displayError(data.error);
            errorSection.style.display = 'block';
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
        errorSection.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
});

function displayResults(result) {
    const isMachine = result.class === 1;
    const badgeClass = isMachine ? 'badge-machine' : 'badge-human';
    const confPercent = (result.confidence * 100).toFixed(1);
    
    resultContent.innerHTML = `
        <div class="result-item">
            <strong>🎤 Prediction</strong>
            <div class="result-value">${result.prediction}</div>
            <span class="prediction-badge ${badgeClass}">
                ${isMachine ? '🤖 Machine Generated' : '👤 Human Voice'}
            </span>
        </div>
        
        <div class="result-item">
            <strong>📊 Confidence Score</strong>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confPercent}%;">
                    ${confPercent}%
                </div>
            </div>
        </div>
        
        <div class="result-item">
            <strong>🎯 Decision</strong>
            <div class="result-value">${result.decision}</div>
        </div>
        
        <div class="result-item">
            <strong>📈 Chunks Analyzed</strong>
            <div class="result-value">${result.chunks_analyzed}</div>
        </div>
    `;
}

function displayError(error) {
    errorMessage.textContent = error;
}

// ==============================
// BATCH UPLOAD HANDLING
// ==============================

const batchUploadArea = document.getElementById('batchUploadArea');
const multipleFileInput = document.getElementById('multipleFileInput');
const selectMultipleBtn = document.getElementById('selectMultipleBtn');
const batchPredictBtn = document.getElementById('batchPredictBtn');
const batchResults = document.getElementById('batchResults');

let selectedFiles = [];

selectMultipleBtn.addEventListener('click', () => {
    multipleFileInput.click();
});

multipleFileInput.addEventListener('change', (e) => {
    selectedFiles = Array.from(e.target.files);
    updateBatchUI();
});

batchUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    batchUploadArea.classList.add('dragover');
});

batchUploadArea.addEventListener('dragleave', () => {
    batchUploadArea.classList.remove('dragover');
});

batchUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    batchUploadArea.classList.remove('dragover');
    
    selectedFiles = Array.from(e.dataTransfer.files);
    multipleFileInput.files = e.dataTransfer.files;
    updateBatchUI();
});

function updateBatchUI() {
    if (selectedFiles.length > 0) {
        batchPredictBtn.disabled = false;
        batchUploadArea.style.borderColor = 'var(--success-color)';
        batchUploadArea.innerHTML = `
            <p>✅ ${selectedFiles.length} file(s) selected</p>
            <p style="color: var(--success-color); margin-top: 10px;">Ready for batch analysis</p>
        `;
    }
}

batchPredictBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    
    batchResults.innerHTML = '';
    batchPredictBtn.disabled = true;
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayBatchResults(data);
        } else {
            batchResults.innerHTML = `<div class="error-message">Error: ${data.error}</div>`;
        }
    } catch (error) {
        batchResults.innerHTML = `<div class="error-message">Network error: ${error.message}</div>`;
    } finally {
        batchPredictBtn.disabled = false;
    }
});

function displayBatchResults(data) {
    let html = `
        <div class="result-card" style="margin-top: 20px;">
            <p><strong>📊 Summary:</strong> ${data.successful} successful, ${data.failed} failed out of ${data.total_files}</p>
    `;
    
    // Successful results
    data.results.forEach(item => {
        const result = item.result;
        const isMachine = result.class === 1;
        const confPercent = (result.confidence * 100).toFixed(1);
        
        html += `
            <div class="batch-result-item success">
                <div>
                    <strong>${item.filename}</strong><br>
                    <span style="color: var(--text-light);">
                        ${result.prediction} (${confPercent}% confidence)
                    </span>
                </div>
                <span style="color: var(--success-color); font-weight: bold;">✅</span>
            </div>
        `;
    });
    
    // Failed results
    data.errors.forEach(item => {
        html += `
            <div class="batch-result-item error">
                <div>
                    <strong>${item.filename}</strong><br>
                    <span style="color: var(--error-color);">${item.error}</span>
                </div>
                <span style="color: var(--error-color); font-weight: bold;">❌</span>
            </div>
        `;
    });
    
    html += '</div>';
    batchResults.innerHTML = html;
}

// ==============================
// PAGE LOAD
// ==============================

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('⚠️ Warning: Model not loaded');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
