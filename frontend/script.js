document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('upload-box');
    const uploadContent = document.getElementById('upload-content');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const getInsightsBtn = document.getElementById('get-insights-btn');
    
    // Sections
    const formsSection = document.getElementById('forms-section');
    const resultsDashboard = document.getElementById('results-dashboard');
    const predictionsSection = document.getElementById('predictions-section');
    const predictionCards = document.getElementById('prediction-cards');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const toastContainer = document.getElementById('toast-container');

    // Result Elements
    const resPrice = document.getElementById('res-price');
    const resTier = document.getElementById('res-tier');
    const resSize = document.getElementById('res-size');
    const resSizeText = document.getElementById('res-size-text');
    const modelBadge = document.getElementById('model-badge');

    let currentCategory = "";
    let selectedFile = null;
    let lastPredictionData = null;

    // ─── Toast Notifications ───
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerText = message;
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            toast.style.transition = 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
            setTimeout(() => toast.remove(), 400);
        }, 3500);
    }

    // ─── Drag & Drop ───
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
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file.', 'error');
            return;
        }
        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            showToast('File too large. Maximum 10MB allowed.', 'error');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
            if (uploadContent) uploadContent.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Analyze Item";
        };
        reader.readAsDataURL(file);
    }

    // ─── Skeleton State Manager ───
    function setSkeletonState(nodes, isSkeleton) {
        nodes.forEach(node => {
            if (!node) return;
            if (isSkeleton) {
                node.classList.add('skeleton');
                if (node.tagName !== 'DIV') {
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

    // ─── Confidence Level ───
    function getConfidenceLevel(confidence) {
        if (confidence >= 0.7) return { level: 'high', label: 'High' };
        if (confidence >= 0.4) return { level: 'medium', label: 'Medium' };
        return { level: 'low', label: 'Low' };
    }

    // ─── Render Top-5 Prediction Cards ───
    function renderPredictionCards(predictions, activeIndex) {
        predictionCards.innerHTML = '';
        
        const topConf = predictions[0]?.confidence || 1;

        predictions.forEach((pred, i) => {
            const pct = (pred.confidence * 100).toFixed(1);
            const isActive = i === activeIndex;
            const { level } = getConfidenceLevel(pred.confidence);
            const barWidth = Math.max(8, (pred.confidence / topConf) * 100);

            const card = document.createElement('div');
            card.className = `pred-card ${isActive ? 'pred-card-active' : ''} pred-card-${level}`;
            card.style.animationDelay = `${i * 0.06}s`;
            card.dataset.index = i;
            card.dataset.category = pred.category;
            card.dataset.confidence = pred.confidence;

            card.innerHTML = `
                <div class="pred-rank">${i + 1}</div>
                <div class="pred-body">
                    <div class="pred-header">
                        <span class="pred-category">${pred.category}</span>
                        ${isActive ? '<span class="pred-selected-badge">✓ Selected</span>' : ''}
                    </div>
                    <div class="pred-bar-track">
                        <div class="pred-bar-fill pred-bar-${level}" style="width: ${barWidth}%"></div>
                    </div>
                    <div class="pred-meta">
                        <span class="pred-confidence">${pct}%</span>
                        <span class="pred-level">${level} confidence</span>
                    </div>
                </div>
            `;

            if (!isActive) {
                card.addEventListener('click', () => {
                    switchToCategory(pred.category, pred.confidence, i);
                });
            }

            predictionCards.appendChild(card);
        });
    }

    // ─── Render Skeleton Cards ───
    function renderSkeletonCards() {
        predictionCards.innerHTML = '';
        for (let i = 0; i < 5; i++) {
            const card = document.createElement('div');
            card.className = 'pred-card pred-card-skeleton';
            card.style.animationDelay = `${i * 0.06}s`;
            card.innerHTML = `
                <div class="pred-rank skeleton-circle"></div>
                <div class="pred-body">
                    <div class="pred-header"><span class="skeleton-text-sm">&nbsp;</span></div>
                    <div class="pred-bar-track"><div class="pred-bar-fill" style="width: ${80 - i * 15}%"></div></div>
                    <div class="pred-meta"><span class="skeleton-text-xs">&nbsp;</span></div>
                </div>
            `;
            predictionCards.appendChild(card);
        }
    }

    // ─── Step 1: Predict ───
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerText = "Analyzing…";

        // Reveal dashboard with skeletons for perceived speed
        formsSection.classList.remove('hidden');
        resultsDashboard.classList.remove('hidden');

        // Show skeleton prediction cards
        renderSkeletonCards();
        modelBadge.textContent = 'Analyzing…';

        // Skeleton placeholders for recommendations
        recommendationsContainer.innerHTML = '';
        for (let i = 0; i < 4; i++) {
            recommendationsContainer.innerHTML += `<div class="img-container skeleton"></div>`;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('https://fashion-api-frtk.onrender.com/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Prediction API returned an error');
            
            const data = await res.json();
            lastPredictionData = data;
            currentCategory = data.category;
            
            // Render prediction cards
            modelBadge.textContent = `Top ${data.top_predictions.length} Candidates`;
            renderPredictionCards(data.top_predictions, 0);
            
            // Populate recommendations
            recommendationsContainer.innerHTML = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach((url, i) => {
                    const div = document.createElement('div');
                    div.className = 'img-container';
                    div.style.animationDelay = `${i * 0.08}s`;
                    div.style.animation = `fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) ${0.1 + i * 0.08}s both`;
                    const img = document.createElement('img');
                    img.src = url.startsWith('/') ? 'https://fashion-api-frtk.onrender.com' + url : url;
                    img.alt = `Similar to ${currentCategory}`;
                    img.loading = 'lazy';
                    img.onerror = function() {
                        this.parentElement.style.display = 'none';
                    };
                    div.appendChild(img);
                    recommendationsContainer.appendChild(div);
                });
            } else {
                recommendationsContainer.innerHTML = '<p class="subtext" style="grid-column:1/-1; text-align:center; padding: 32px 0;">No visually similar matches found.</p>';
            }

            // Auto-calculate insights
            fetchInsights();
            showToast(`Detected: ${currentCategory}`, 'success');

            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Analyze Item";

        } catch (error) {
            showToast('Analysis failed — ' + error.message, 'error');
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Retry Analysis";
        }
    });

    // ─── Switch Category (user correction) ───
    async function switchToCategory(newCategory, confidence, activeIndex) {
        if (newCategory === currentCategory) return;

        const originalCategory = currentCategory;
        const originalConfidence = lastPredictionData ? lastPredictionData.confidence : 0;
        currentCategory = newCategory;

        // Re-render prediction cards with new active state
        if (lastPredictionData && lastPredictionData.top_predictions) {
            renderPredictionCards(lastPredictionData.top_predictions, activeIndex);
        }

        // Send feedback correction to backend
        if (selectedFile) {
            try {
                const feedbackForm = new FormData();
                feedbackForm.append('file', selectedFile);
                feedbackForm.append('original_category', originalCategory);
                feedbackForm.append('corrected_category', newCategory);
                feedbackForm.append('original_confidence', originalConfidence);

                const fbRes = await fetch('https://fashion-api-frtk.onrender.com/feedback', { method: 'POST', body: feedbackForm });
                const fbData = await fbRes.json();
                if (fbData.status === 'saved') {
                    showToast(`✓ Correction saved (${fbData.total_corrections} total)`, 'success');
                } else {
                    showToast(`Switched to: ${currentCategory}`, 'info');
                }
            } catch (e) {
                console.error('Feedback submission failed:', e);
                showToast(`Switched to: ${currentCategory}`, 'info');
            }
        } else {
            showToast(`Switched to: ${currentCategory}`, 'info');
        }

        // Show updating state
        analyzeBtn.innerText = "Updating…";
        analyzeBtn.disabled = true;

        // Re-fetch recommendations for the corrected category
        if (selectedFile) {
            recommendationsContainer.innerHTML = '';
            for (let i = 0; i < 4; i++) {
                recommendationsContainer.innerHTML += `<div class="img-container skeleton"></div>`;
            }

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('brand', document.getElementById('brand').value || '');

                const res = await fetch('https://fashion-api-frtk.onrender.com/predict', { method: 'POST', body: formData });
                const data = await res.json();

                // Use recommendations from the response (they're based on visual similarity)
                recommendationsContainer.innerHTML = '';
                if (data.recommendations && data.recommendations.length > 0) {
                    data.recommendations.forEach((url, i) => {
                        const div = document.createElement('div');
                        div.className = 'img-container';
                        div.style.animation = `fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) ${0.1 + i * 0.08}s both`;
                        const img = document.createElement('img');
                        img.src = url.startsWith('/') ? 'https://fashion-api-frtk.onrender.com' + url : url;
                        img.alt = `Similar to ${currentCategory}`;
                        img.loading = 'lazy';
                        img.onerror = function() { this.parentElement.style.display = 'none'; };
                        div.appendChild(img);
                        recommendationsContainer.appendChild(div);
                    });
                } else {
                    recommendationsContainer.innerHTML = '<p class="subtext" style="grid-column:1/-1; text-align:center; padding: 32px 0;">No visually similar matches found.</p>';
                }
            } catch (e) {
                console.error('Failed to refresh recommendations:', e);
            } finally {
                analyzeBtn.innerText = "Analyze Item";
                analyzeBtn.disabled = false;
            }
        } else {
            analyzeBtn.innerText = "Analyze Item";
            analyzeBtn.disabled = false;
        }

        // Re-fetch insights for the corrected category
        fetchInsights();
    }

    // ─── Step 2: Insights (Resale + Size) ───
    getInsightsBtn.addEventListener('click', fetchInsights);

    async function fetchInsights() {
        if (!currentCategory) return;
        
        const brand = document.getElementById('brand').value;
        const condition = document.getElementById('condition').value;
        
        const chest = parseFloat(document.getElementById('chest').value);
        const waist = parseFloat(document.getElementById('waist').value);
        const hip = parseFloat(document.getElementById('hip').value);

        getInsightsBtn.disabled = true;
        getInsightsBtn.innerText = "Calculating…";

        setSkeletonState([resPrice, resSize], true);
        resTier.innerHTML = '<span class="pulse-dot"></span> Evaluating market data…';
        resSizeText.innerHTML = '<span class="pulse-dot"></span> Comparing brand geometries…';

        try {
            // Resale Value
            const resaleRes = await fetch('https://fashion-api-frtk.onrender.com/estimate-resale', {
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
                resPrice.innerText = `$${resaleData.min_price} – $${resaleData.max_price}`;
                resTier.innerText = `${condition} condition • ${resaleData.tier} tier`;
            }

            // Size Translation
            const sizePayload = { target_brand: brand, category: currentCategory };
            if (!isNaN(chest)) {
                sizePayload.chest = chest;
                sizePayload.bust = chest; // Map to bust for dresses
            }
            if (!isNaN(waist)) sizePayload.waist = waist;
            if (!isNaN(hip)) sizePayload.hip = hip;

            const sizeRes = await fetch('https://fashion-api-frtk.onrender.com/translate-size', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sizePayload)
            });
            const sizeData = await sizeRes.json();
            
            setSkeletonState([resSize], false);
            if (sizeData.error) {
                resSize.innerText = 'N/A';
                resSizeText.innerText = "Enter measurements for sizing";
            } else {
                resSize.innerText = sizeData.recommended_size;
                resSizeText.innerText = `Best fit for ${brand}`;
            }

        } catch (error) {
            console.error(error);
            showToast('Insight calculation failed', 'error');
        } finally {
            getInsightsBtn.disabled = false;
            getInsightsBtn.innerText = "Recalculate Insights";
        }
    }
});
