document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Selection with safe fallback capability
    const chatFeed = document.getElementById('chat-feed');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const btnSend = document.getElementById('btn-send');
    const btnIngest = document.getElementById('btn-ingest');
    const statEssays = document.getElementById('stat-essays');
    const statChunks = document.getElementById('stat-chunks');
    const progressContainer = document.getElementById('progress-container');
    const progressStatusText = document.getElementById('progress-status-text');
    const progressPercent = document.getElementById('progress-percent');
    const progressFill = document.getElementById('progress-fill');
    const statusErrorMsg = document.getElementById('status-error-msg');
    const essaysList = document.getElementById('essays-list');
    const librarySearchInput = document.getElementById('library-search-input');
    const engineBadge = document.getElementById('engine-badge');
    const pulseIndicator = document.querySelector('.pulse-indicator');
    const welcomeSamples = document.getElementById('welcome-samples');

    // Safe DOM Mutation Helper functions to fully prevent null property access errors
    function safeSetText(el, text) {
        if (el) el.textContent = text;
    }

    function safeSetHTML(el, html) {
        if (el) el.innerHTML = html;
    }

    function safeSetStyleDisplay(el, displayValue) {
        if (el) el.style.display = displayValue;
    }

    function safeSetStyleWidth(el, widthValue) {
        if (el) el.style.width = widthValue;
    }

    function safeSetDisabled(el, isDisabled) {
        if (el) el.disabled = isDisabled;
    }

    function safeSetPlaceholder(el, placeholderText) {
        if (el) el.placeholder = placeholderText;
    }

    function safeSetValue(el, valueText) {
        if (el) el.value = valueText;
    }

    function safeFocus(el) {
        if (el) el.focus();
    }

    // Application State
    let isIngesting = false;
    let pollingInterval = null;
    let essaysData = []; // Cache list of essays for search filtering
    
    // Initialize Status
    checkStatus();
    loadEssays();
    
    // ----------------------------------------------------
    // API Function calls
    // ----------------------------------------------------
    
    async function checkStatus() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) throw new Error("Failed to fetch engine status");
            const data = await response.json();
            
            // Update stats counter safely
            safeSetText(statEssays, data.total_essays);
            safeSetText(statChunks, data.total_chunks);
            
            // Manage ingestion state machine
            if (data.status === 'scraping' || data.status === 'embedding' || data.status === 'starting') {
                isIngesting = true;
                setIngestionUIState(true);
                updateProgressBar(data.status, data.progress);
                setEngineBadge('Indexing...', 'busy');
                
                // Allow chat while indexing if we already have chunks!
                if (data.total_chunks > 0) {
                    safeSetDisabled(chatInput, false);
                    safeSetDisabled(btnSend, false);
                    safeSetPlaceholder(chatInput, "Ask a question about Paul Graham's essays...");
                    safeSetStyleDisplay(welcomeSamples, 'block');
                } else {
                    safeSetDisabled(chatInput, true);
                    safeSetDisabled(btnSend, true);
                    safeSetPlaceholder(chatInput, "Please wait for database indexing to begin...");
                    safeSetStyleDisplay(welcomeSamples, 'none');
                }
                
                // Start polling if not already active
                if (!pollingInterval) {
                    pollingInterval = setInterval(pollIngestionStatus, 1500);
                }
            } else if (data.status === 'done' || data.total_chunks > 0) {
                isIngesting = false;
                setIngestionUIState(false);
                setEngineBadge('Engine Online', 'active');
                safeSetDisabled(chatInput, false);
                safeSetDisabled(btnSend, false);
                safeSetPlaceholder(chatInput, "Ask a question about Paul Graham's essays...");
                safeSetStyleDisplay(welcomeSamples, 'block');
                
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                }
            } else {
                // Database is completely empty / idle
                isIngesting = false;
                setIngestionUIState(false);
                setEngineBadge('Engine Offline', 'inactive');
                safeSetDisabled(chatInput, true);
                safeSetDisabled(btnSend, true);
                safeSetPlaceholder(chatInput, "Please index essays first to enable chatbot...");
                safeSetStyleDisplay(welcomeSamples, 'none');
            }
        } catch (error) {
            console.error(error);
            setEngineBadge('Connection Error', 'inactive');
        }
    }
    
    async function loadEssays() {
        try {
            const response = await fetch('/api/essays');
            if (!response.ok) return;
            const data = await response.json();
            essaysData = data.essays;
            renderEssaysList(essaysData);
        } catch (error) {
            console.error("Failed to load essay index:", error);
        }
    }
    
    async function triggerIngestion() {
        try {
            safeSetStyleDisplay(statusErrorMsg, 'none');
            const response = await fetch('/api/ingest', { method: 'POST' });
            const data = await response.json();
            
            // Set UI and initiate polling
            isIngesting = true;
            setIngestionUIState(true);
            updateProgressBar('starting', 0);
            setEngineBadge('Connecting...', 'busy');
            
            if (!pollingInterval) {
                pollingInterval = setInterval(pollIngestionStatus, 1500);
            }
        } catch (error) {
            safeSetText(statusErrorMsg, "Could not trigger ingestion pipeline. Please try again.");
            safeSetStyleDisplay(statusErrorMsg, 'block');
            setIngestionUIState(false);
        }
    }
    
    async function pollIngestionStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            safeSetText(statEssays, data.total_essays);
            safeSetText(statChunks, data.total_chunks);
            
            updateProgressBar(data.status, data.progress);
            
            // Allow chat while indexing if we already have chunks!
            if (data.total_chunks > 0 && chatInput && chatInput.disabled) {
                safeSetDisabled(chatInput, false);
                safeSetDisabled(btnSend, false);
                safeSetPlaceholder(chatInput, "Ask a question about Paul Graham's essays...");
                safeSetStyleDisplay(welcomeSamples, 'block');
            }
            
            if (data.status === 'done') {
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                }
                isIngesting = false;
                
                setIngestionUIState(false);
                setEngineBadge('Engine Online', 'active');
                
                // Enable Chat Inputs
                safeSetDisabled(chatInput, false);
                safeSetDisabled(btnSend, false);
                safeSetPlaceholder(chatInput, "Ask a question about Paul Graham's essays...");
                safeSetStyleDisplay(welcomeSamples, 'block');
                
                // Refresh Essays list
                loadEssays();
                
                // Show completion message in chat
                appendSystemMessage("Database indexing complete! " + data.total_essays + " essays successfully parsed, chunked, and vector-embedded. You can now chat.");
            } else if (data.status === 'error') {
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                }
                isIngesting = false;
                
                setIngestionUIState(false);
                setEngineBadge('Engine Error', 'inactive');
                
                safeSetText(statusErrorMsg, `Ingestion error: ${data.error}`);
                safeSetStyleDisplay(statusErrorMsg, 'block');
            }
        } catch (error) {
            console.error("Error polling ingestion status:", error);
        }
    }
    
    async function sendChatMessage(query) {
        // Disable inputs safely
        safeSetDisabled(chatInput, true);
        safeSetDisabled(btnSend, true);
        
        // Show Typing Indicator
        const typingEl = appendTypingIndicator();
        if (chatFeed) chatFeed.scrollTop = chatFeed.scrollHeight;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: query })
            });
            
            // Remove typing indicator safely
            if (typingEl) typingEl.remove();
            
            if (!response.ok) {
                let errorMsg = "Server error";
                try {
                    const errData = await response.json();
                    errorMsg = errData.detail || errorMsg;
                } catch (e) {
                    // Fallback to reading response as plain text if it is not JSON
                    try {
                        const text = await response.text();
                        if (text && text.length < 200) errorMsg = text;
                    } catch (_) {}
                }
                throw new Error(errorMsg);
            }
            
            const data = await response.json();
            
            // Render Answer
            appendBotMessage(data.answer, data.sources);
            
        } catch (error) {
            if (typingEl) typingEl.remove();
            appendBotMessage(`⚠️ An error occurred: ${error.message}. Please check your environment variables (GEMINI_API_KEY) or system logs.`, []);
        } finally {
            safeSetDisabled(chatInput, false);
            safeSetDisabled(btnSend, false);
            safeSetValue(chatInput, '');
            safeFocus(chatInput);
            if (chatFeed) chatFeed.scrollTop = chatFeed.scrollHeight;
        }
    }
    
    // ----------------------------------------------------
    // UI Rendering helpers
    // ----------------------------------------------------
    
    function setIngestionUIState(active) {
        if (btnIngest) {
            btnIngest.disabled = active;
            if (active) {
                btnIngest.innerHTML = `
                    <svg class="icon spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="2" x2="12" y2="6"></line>
                        <line x1="12" y1="18" x2="12" y2="22"></line>
                        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                        <line x1="2" y1="12" x2="6" y2="12"></line>
                        <line x1="18" y1="12" x2="22" y2="12"></line>
                        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
                        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
                    </svg>
                    <span>Indexing...</span>
                `;
            } else {
                btnIngest.innerHTML = `
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                    </svg>
                    <span>Index Essays</span>
                `;
            }
        }
        safeSetStyleDisplay(progressContainer, active ? 'block' : 'none');
    }
    
    function updateProgressBar(status, progress) {
        safeSetStyleWidth(progressFill, `${progress}%`);
        safeSetText(progressPercent, `${progress}%`);
        
        if (progressStatusText) {
            if (status === 'starting') {
                progressStatusText.textContent = "Booting pipeline...";
            } else if (status === 'scraping') {
                progressStatusText.textContent = "Crawling essays...";
            } else if (status === 'scraping_done') {
                progressStatusText.textContent = "Essays scraped. Chunking...";
            } else if (status === 'embedding') {
                progressStatusText.textContent = "Generating vector embeddings...";
            } else if (status === 'done') {
                progressStatusText.textContent = "Indexing successful!";
            }
        }
    }
    
    function setEngineBadge(text, type) {
        safeSetText(engineBadge, text);
        if (pulseIndicator) {
            pulseIndicator.className = 'pulse-indicator'; // Reset classes
            if (type) pulseIndicator.classList.add(type);
        }
    }
    
    function renderEssaysList(essays) {
        if (!essaysList) return;
        essaysList.innerHTML = '';
        if (essays.length === 0) {
            essaysList.innerHTML = '<div class="empty-library">No essays loaded yet. Click "Index Essays" to scrape Paul Graham\'s articles.</div>';
            return;
        }
        
        essays.forEach(essay => {
            const a = document.createElement('a');
            a.href = essay.url;
            a.target = '_blank';
            a.className = 'essay-item';
            a.innerHTML = `
                <span>${essay.title}</span>
                <svg class="essay-link-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
            `;
            essaysList.appendChild(a);
        });
    }
    
    function appendSystemMessage(text) {
        if (!chatFeed) return;
        const div = document.createElement('div');
        div.className = 'message system-msg glass fade-in';
        div.innerHTML = `
            <div class="msg-avatar">
                <div class="avatar-glow"></div>
                <svg class="avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12c0-2.4 1.35-4.7 3.5-6"/>
                </svg>
            </div>
            <div class="msg-bubble">
                <p>${text}</p>
            </div>
        `;
        chatFeed.appendChild(div);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }
    
    function appendUserMessage(text) {
        if (!chatFeed) return;
        const div = document.createElement('div');
        div.className = 'message user fade-in';
        div.innerHTML = `
            <div class="msg-avatar">
                <div class="avatar-glow"></div>
                <svg class="avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <div class="msg-bubble">
                <p>${escapeHtml(text)}</p>
            </div>
        `;
        chatFeed.appendChild(div);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }
    
    function appendBotMessage(markdownText, sources) {
        if (!chatFeed) return;
        const div = document.createElement('div');
        div.className = 'message bot fade-in';
        
        // Format basic markdown elements safely
        let formattedText = formatMarkdown(markdownText);
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="source-pill-container">
                    <span class="source-label">Cited Essays:</span>
                    ${sources.map((s, idx) => `
                        <a href="${s.url}" target="_blank" class="source-pill">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                            </svg>
                            <span>${s.title}</span>
                        </a>
                    `).join('')}
                </div>
            `;
        }
        
        div.innerHTML = `
            <div class="msg-avatar">
                <div class="avatar-glow"></div>
                <svg class="avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M12 2v9"></path>
                    <path d="M8 5h8"></path>
                </svg>
            </div>
            <div class="msg-bubble">
                ${formattedText}
                ${sourcesHtml}
            </div>
        `;
        
        chatFeed.appendChild(div);
        chatFeed.scrollTop = chatFeed.scrollHeight;
        
        // Add click events to citation links in text to open corresponding source pill link
        div.querySelectorAll('.citation').forEach(cite => {
            cite.addEventListener('click', (e) => {
                e.preventDefault();
                const index = parseInt(cite.dataset.index);
                if (sources && sources[index - 1]) {
                    window.open(sources[index - 1].url, '_blank');
                }
            });
        });
    }
    
    function appendTypingIndicator() {
        if (!chatFeed) return null;
        const div = document.createElement('div');
        div.className = 'message bot fade-in';
        div.innerHTML = `
            <div class="msg-avatar">
                <div class="avatar-glow"></div>
                <svg class="avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M12 2v9"></path>
                    <path d="M8 5h8"></path>
                </svg>
            </div>
            <div class="msg-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatFeed.appendChild(div);
        return div;
    }
    
    // ----------------------------------------------------
    // Event Listeners - Guarded to handle missing elements safely
    // ----------------------------------------------------
    
    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (!chatInput) return;
            const text = chatInput.value.trim();
            if (!text) return;
            
            appendUserMessage(text);
            safeSetValue(chatInput, ''); // Clear input instantly
            sendChatMessage(text);
        });
    }
    
    if (btnIngest) {
        btnIngest.addEventListener('click', () => {
            if (isIngesting) return;
            triggerIngestion();
        });
    }
    
    // Sidebar search filter (instant responsive search)
    if (librarySearchInput) {
        librarySearchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderEssaysList(essaysData);
                return;
            }
            
            const filtered = essaysData.filter(essay => 
                essay.title.toLowerCase().includes(query)
            );
            renderEssaysList(filtered);
        });
    }
    
    // Sample Queries Chips click action
    if (chatFeed) {
        chatFeed.addEventListener('click', (e) => {
            if (e.target.classList.contains('chip')) {
                const query = e.target.textContent;
                safeSetValue(chatInput, query);
                appendUserMessage(query);
                safeSetValue(chatInput, ''); // Clear input instantly
                sendChatMessage(query);
            }
        });
    }
    
    // ----------------------------------------------------
    // Utilities
    // ----------------------------------------------------
    
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function formatMarkdown(text) {
        if (!text) return '';
        // Convert citations: [1], [2] to interactive link tags
        let formatted = text.replace(/\[(\d+)\]/g, (match, num) => {
            return `<a href="#" class="citation" data-index="${num}">[${num}]</a>`;
        });
        
        const paras = formatted.split('\n\n');
        
        return paras.map(p => {
            p = p.trim();
            if (!p) return '';
            
            // Check list items
            if (p.startsWith('- ') || p.startsWith('* ')) {
                const items = p.split(/\n[*-]\s+/);
                // First item starts with "- ", strip it
                items[0] = items[0].replace(/^[*-]\s+/, '');
                return `<ul>${items.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join('')}</ul>`;
            }
            
            if (p.match(/^\d+\.\s+/)) {
                const items = p.split(/\n\d+\.\s+/);
                items[0] = items[0].replace(/^\d+\.\s+/, '');
                return `<ol>${items.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join('')}</ol>`;
            }
            
            return `<p>${formatInlineMarkdown(p)}</p>`;
        }).join('');
    }
    
    function formatInlineMarkdown(text) {
        if (!text) return '';
        // Simple bold parser
        let inline = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Simple italic parser
        inline = inline.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Simple inline code parser
        inline = inline.replace(/`(.*?)`/g, '<code>$1</code>');
        return inline;
    }
});
