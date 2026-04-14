document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const getInsightsBtn = document.getElementById('get-insights-btn');
    const spinner = document.getElementById('loading-spinner');
    
    // Sections
    const formsSection = document.getElementById('forms-section');
    const resultsDashboard = document.getElementById('results-dashboard');
    const recommendationsContainer = document.getElementById('recommendations-container');

    let currentCategory = "";
    let selectedFile = null;

    // Drag and Drop Logic
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
        };
        reader.readAsDataURL(file);
    }

    // Step 1: Predict Category
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        spinner.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Prediction failed');
            
            const data = await res.json();
            currentCategory = data.category;
            
            // Show base results
            document.getElementById('res-category').innerText = currentCategory;
            document.getElementById('res-confidence').innerText = 'AI Detection Confirmed';
            
            // Show recommendations
            recommendationsContainer.innerHTML = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(url => {
                    const div = document.createElement('div');
                    div.className = 'img-container';
                    div.innerHTML = `<img src="${url}" alt="Similar item" loading="lazy">`;
                    recommendationsContainer.appendChild(div);
                });
            } else {
                recommendationsContainer.innerHTML = '<p class="subtext">No similar items found in dataset.</p>';
            }

            // Reveal next steps
            formsSection.classList.remove('hidden');
            resultsDashboard.classList.remove('hidden');

            // Optionally, automatically fetch insights right away using defaults
            fetchInsights();

        } catch (error) {
            alert('Error analyzing image: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    // Step 2: Get Insights
    getInsightsBtn.addEventListener('click', fetchInsights);

    async function fetchInsights() {
        if (!currentCategory) return;
        
        const brand = document.getElementById('brand').value;
        const condition = document.getElementById('condition').value;
        
        const chest = parseFloat(document.getElementById('chest').value);
        const waist = parseFloat(document.getElementById('waist').value);
        const hip = parseFloat(document.getElementById('hip').value);

        getInsightsBtn.disabled = true;
        getInsightsBtn.innerText = "Analyzing...";

        try {
            // Fetch Resale Value
            const resaleRes = await fetch('/estimate-resale', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brand, category: currentCategory, condition })
            });
            const resaleData = await resaleRes.json();
            if (resaleData.error) {
                document.getElementById('res-price').innerText = 'N/A';
                document.getElementById('res-tier').innerText = resaleData.error;
            } else {
                document.getElementById('res-price').innerText = `$${resaleData.min_price} - $${resaleData.max_price}`;
                document.getElementById('res-tier').innerText = `Est. Value (${resaleData.tier})`;
            }

            // Fetch Size Translation
            const sizePayload = { target_brand: brand, category: currentCategory };
            if (!isNaN(chest)) sizePayload.chest = chest;
            if (!isNaN(waist)) sizePayload.waist = waist;
            if (!isNaN(hip)) sizePayload.hip = hip;

            // Only attempt sizing if we have some measurements to avoid bad requests if you need them.
            // But the backend handles missing, so let's send it.
            const sizeRes = await fetch('/translate-size', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sizePayload)
            });
            const sizeData = await sizeRes.json();
            
            if (sizeData.error) {
                document.getElementById('res-size').innerText = 'N/A';
            } else {
                document.getElementById('res-size').innerText = sizeData.recommended_size;
            }

        } catch (error) {
            console.error(error);
            alert('Error getting insights: ' + error.message);
        } finally {
            getInsightsBtn.disabled = false;
            getInsightsBtn.innerText = "Get AI Insights";
        }
    }
});
