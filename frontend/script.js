document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const getInsightsBtn = document.getElementById('get-insights-btn');
    
    // Elements
    const formsSection = document.getElementById('forms-section');
    const resultsDashboard = document.getElementById('results-dashboard');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const toastContainer = document.getElementById('toast-container');

    // Result Nodes
    const resCategory = document.getElementById('res-category');
    const resConfidence = document.getElementById('res-confidence');
    const resPrice = document.getElementById('res-price');
    const resTier = document.getElementById('res-tier');
    const resSize = document.getElementById('res-size');

    let currentCategory = "";
    let selectedFile = null;

    // --- Toast Notifications ---
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerText = message;
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // --- Drag and Drop Logic ---
    uploadBox.addEventListener('click', () => fileInput.click());
    
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Analyze Item";
        };
        reader.readAsDataURL(file);
    }

    // --- Helper: Skeleton State ---
    function setSkeletonState(nodes, isSkeleton) {
        nodes.forEach(node => {
            if (!node) return;
            if (isSkeleton) {
                node.classList.add('skeleton');
                if (node.tagName !== 'DIV') {
                    // Cache old text if not already cached
                    if (!node.dataset.originalText) {
                        node.dataset.originalText = node.innerText;
                    }
                    node.innerText = "---";
                }
            } else {
                node.classList.remove('skeleton');
            }
        });
    }

    // --- Step 1: Predict ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerText = "Analyzing...";

        // Show dashboard with skeletons immediately for perceived performance
        formsSection.classList.remove('hidden');
        resultsDashboard.classList.remove('hidden');

        // Set predict skeletons
        setSkeletonState([resCategory], true);
        resConfidence.innerText = "Running AI analysis...";

        // Set predict image skeletons
        recommendationsContainer.innerHTML = '';
        for(let i=0; i<4; i++) {
            recommendationsContainer.innerHTML += `<div class="img-container skeleton"></div>`;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Prediction API failed');
            
            const data = await res.json();
            currentCategory = data.category;
            
            // Remove skeletons & apply text
            setSkeletonState([resCategory], false);
            resCategory.innerText = currentCategory;
            resConfidence.innerText = 'Detection Confirmed';
            
            // Render similar images
            recommendationsContainer.innerHTML = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(url => {
                    const div = document.createElement('div');
                    div.className = 'img-container';
                    div.innerHTML = `<img src="${url}" alt="Similar style" loading="lazy">`;
                    recommendationsContainer.appendChild(div);
                });
            } else {
                recommendationsContainer.innerHTML = '<p class="subtext">No highly similar matches found.</p>';
            }

            // Move right into calculating pricing/sizing
            fetchInsights();

        } catch (error) {
            showToast('Error analyzing image: ' + error.message);
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Retry Analysis";
        }
    });

    // --- Step 2: Insights ---
    getInsightsBtn.addEventListener('click', fetchInsights);

    async function fetchInsights() {
        if (!currentCategory) return;
        
        const brand = document.getElementById('brand').value;
        const condition = document.getElementById('condition').value;
        
        const chest = parseFloat(document.getElementById('chest').value);
        const waist = parseFloat(document.getElementById('waist').value);
        const hip = parseFloat(document.getElementById('hip').value);

        getInsightsBtn.disabled = true;
        getInsightsBtn.innerText = "Calculating...";

        setSkeletonState([resPrice, resSize], true);
        resTier.innerText = "Evaluating market data...";
        document.getElementById('res-size-text').innerText = "Comparing brand geometries...";

        try {
            // Fetch Resale Value
            const resaleRes = await fetch('/estimate-resale', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brand, category: currentCategory, condition })
            });
            const resaleData = await resaleRes.json();
            
            setSkeletonState([resPrice], false);
            if (resaleData.error) {
                resPrice.innerText = 'N/A';
                resTier.innerText = resaleData.error;
            } else {
                // Formatting values cleanly
                resPrice.innerText = `$${resaleData.min_price} - $${resaleData.max_price}`;
                resTier.innerText = `Condition: ${condition} • ${resaleData.tier}`;
            }

            // Fetch Size Translation
            const sizePayload = { target_brand: brand, category: currentCategory };
            if (!isNaN(chest)) sizePayload.chest = chest;
            if (!isNaN(waist)) sizePayload.waist = waist;
            if (!isNaN(hip)) sizePayload.hip = hip;

            const sizeRes = await fetch('/translate-size', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sizePayload)
            });
            const sizeData = await sizeRes.json();
            
            setSkeletonState([resSize], false);
            if (sizeData.error) {
                resSize.innerText = 'N/A';
                document.getElementById('res-size-text').innerText = "Provide measurements to unlock";
            } else {
                resSize.innerText = sizeData.recommended_size;
                document.getElementById('res-size-text').innerText = `Best fit for ${brand}`;
            }

        } catch (error) {
            console.error(error);
            showToast('Insight calculation failed: ' + error.message);
        } finally {
            getInsightsBtn.disabled = false;
            getInsightsBtn.innerText = "Recalculate Insights";
        }
    }
});
