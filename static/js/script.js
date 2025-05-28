class CodeOptimizer {
            constructor() {
                this.initializeElements();
                this.attachEventListeners();
                this.updatePlaceholders();
            }

            initializeElements() {
                this.languageSelect = document.getElementById('language');
                this.optimizeBtn = document.getElementById('optimize-btn');
                this.inputCode = document.getElementById('input-code');
                this.outputCode = document.getElementById('output-code');
                this.statusDot = document.getElementById('status-dot');
                this.statusText = document.getElementById('status-text');
                this.statusInfo = document.getElementById('status-info');
                this.loadingOverlay = document.getElementById('loading-overlay');
                this.errorMessage = document.getElementById('error-message');
                this.metrics = document.getElementById('metrics');
            }

            attachEventListeners() {
                this.optimizeBtn.addEventListener('click', () => this.optimizeCode());
                this.languageSelect.addEventListener('change', () => this.updatePlaceholders());
                
                // Auto-resize textareas
                [this.inputCode, this.outputCode].forEach(textarea => {
                    textarea.addEventListener('input', () => this.autoResize(textarea));
                });
            }

            updatePlaceholders() {
                const language = this.languageSelect.value;
                const placeholders = {
                    c: `Enter C or Python Code to optimize.`,
                    python: `Enter C or Python Code to optimize.`
                };
                
                if (this.inputCode.value.trim() === '' || this.inputCode.value === this.inputCode.getAttribute('placeholder')) {
                    this.inputCode.value = placeholders[language];
                }
                this.inputCode.setAttribute('placeholder', `Enter your ${language.toUpperCase()} code here...`);
            }

            autoResize(textarea) {
                textarea.style.height = 'auto';
                textarea.style.height = Math.max(textarea.scrollHeight, 200) + 'px';
            }

            getSelectedOptimizations() {
                const optimizations = [];
                const checkboxes = [
                    'dead-code', 'loop-opt', 'constant-fold', 
                    'strength-reduction', 'function-inline'
                ];
                
                checkboxes.forEach(id => {
                    if (document.getElementById(id).checked) {
                        optimizations.push(id.replace('-', '_'));
                    }
                });
                
                return optimizations;
            }

            setStatus(type, text, info = '') {
                this.statusDot.className = `status-dot ${type}`;
                this.statusText.textContent = text;
                this.statusInfo.textContent = info;
            }

            showError(message) {
                this.errorMessage.textContent = message;
                this.errorMessage.style.display = 'block';
                setTimeout(() => {
                    this.errorMessage.style.display = 'none';
                }, 5000);
            }

            showLoading(show = true) {
                this.loadingOverlay.style.display = show ? 'flex' : 'none';
                this.optimizeBtn.disabled = show;
                
                if (show) {
                    this.setStatus('processing', 'Processing', 'Optimizing your code...');
                }
            }

            updateMetrics(data) {
                if (data.metrics) {
                    document.getElementById('lines-removed').textContent = data.metrics.lines_removed || 0;
                    document.getElementById('optimizations-applied').textContent = data.metrics.optimizations_applied || 0;
                    document.getElementById('processing-time').textContent = `${data.metrics.processing_time || 0}ms`;
                    document.getElementById('size-reduction').textContent = `${data.metrics.size_reduction || 0}%`;
                    this.metrics.style.display = 'grid';
                }
            }

            async optimizeCode() {
                const code = this.inputCode.value.trim();
                if (!code) {
                    this.showError('Please enter some code to optimize.');
                    return;
                }

                this.showLoading(true);

                try {
                    const response = await fetch('/optimize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            code: code,
                            language: this.languageSelect.value,
                            optimizations: this.getSelectedOptimizations()
                        })
                    });

                    const data = await response.json();

                    if (data.success) {
                        this.outputCode.value = data.optimized_code;
                        this.setStatus('success', 'Optimization Complete', `Applied ${data.optimizations_applied || 0} optimizations`);
                        this.updateMetrics(data);
                    } else {
                        this.showError(data.error || 'Optimization failed');
                        this.setStatus('error', 'Optimization Failed', data.error || 'Unknown error');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    this.showError('Network error: Could not connect to optimization server');
                    this.setStatus('error', 'Network Error', 'Could not connect to server');
                } finally {
                    this.showLoading(false);
                }
            }
        }

        // Initialize the application when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => {
            new CodeOptimizer();
        });