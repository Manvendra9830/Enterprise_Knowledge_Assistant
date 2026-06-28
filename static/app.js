/**
 * NovaTech Enterprise Knowledge Assistant
 * Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let sessionId = crypto.randomUUID();
    let isWaitingForResponse = false;
    let documentToDelete = null;
    
    // --- DOM Elements ---
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const sendBtn = document.getElementById('send-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    const docList = document.getElementById('document-list');
    const fileUpload = document.getElementById('file-upload');
    const uploadStatus = document.getElementById('upload-status');
    const sourceTemplate = document.getElementById('source-template');
    const ollamaStatus = document.getElementById('ollama-status');

    // Modal & Toast Elements
    const deleteModal = document.getElementById('delete-modal');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const modalDeleteBtn = document.getElementById('modal-delete-btn');
    const deleteModalText = document.getElementById('delete-modal-text');
    const toastNotification = document.getElementById('toast-notification');

    // --- Initialization ---
    checkHealth();
    loadDocuments();
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value.trim() === '') {
            this.style.height = 'auto';
        }
    });

    // Handle Enter key (submit on Enter, newline on Shift+Enter)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (userInput.value.trim() && !isWaitingForResponse) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // New Chat
    newChatBtn.addEventListener('click', () => {
        sessionId = crypto.randomUUID();
        // Keep only the first welcome message
        const welcome = chatHistory.firstElementChild;
        chatHistory.innerHTML = '';
        chatHistory.appendChild(welcome);
        userInput.value = '';
        userInput.focus();
    });

    // --- Chat Submission ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const question = userInput.value.trim();
        if (!question || isWaitingForResponse) return;
        
        // 1. Add user message to UI
        addMessage(question, 'user');
        
        // 2. Reset input
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // 3. Show typing indicator
        isWaitingForResponse = true;
        sendBtn.disabled = true;
        const typingId = showTypingIndicator();
        
        try {
            // 4. Call API
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId })
            });
            
            if (!response.ok) {
                throw new Error(response.status === 429 ? "quota_exhausted" : 'api_error');
            }
            
            const data = await response.json();
            
            // 5. Remove typing indicator & show response
            removeMessage(typingId);
            addMessage(data.answer, 'assistant', data);
            
        } catch (error) {
            console.error('Chat error:', error);
            removeMessage(typingId);
            
            let errMsg = 'Sorry, I encountered an error while processing your request. Please try again.';
            if (error.message === 'quota_exhausted') {
                 errMsg = "The primary language model is temporarily unavailable. The system automatically switched to a fallback model, but it seems there was an issue. Please try again.";
            }
            addMessage(errMsg, 'assistant', { confidence_level: 'none' });
        } finally {
            isWaitingForResponse = false;
            sendBtn.disabled = false;
            userInput.focus();
            
            // Re-check health just in case models changed state
            checkHealth();
        }
    });

    // --- Document Upload ---
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        uploadStatus.textContent = 'Uploading and processing...';
        uploadStatus.style.color = 'var(--text-muted)';
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            
            uploadStatus.textContent = `Success! Processed ${data.chunks_processed} chunks.`;
            uploadStatus.style.color = 'var(--conf-high)';
            
            // Refresh document list
            loadDocuments();
            showToast('Document uploaded successfully!');
            
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = 'Error uploading document.';
            uploadStatus.style.color = 'var(--conf-none)';
        } finally {
            setTimeout(() => { uploadStatus.textContent = ''; }, 5000);
            e.target.value = ''; // Reset input
        }
    });

    // --- Modal Logic ---
    modalCancelBtn.addEventListener('click', () => {
        deleteModal.style.display = 'none';
        documentToDelete = null;
    });

    modalDeleteBtn.addEventListener('click', async () => {
        if (!documentToDelete) return;
        
        const docName = documentToDelete;
        deleteModal.style.display = 'none';
        
        try {
            const response = await fetch(`/api/documents/${encodeURIComponent(docName)}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                loadDocuments();
                showToast(`Document "${docName}" deleted successfully.`);
            } else {
                showToast(`Failed to delete document: ${docName}`);
            }
        } catch (error) {
            console.error('Delete error:', error);
            showToast(`Error deleting document: ${docName}`);
        } finally {
            documentToDelete = null;
        }
    });

    // --- Helper Functions ---

    function showToast(message) {
        toastNotification.textContent = message;
        toastNotification.style.display = 'block';
        
        // Trigger reflow for transition
        toastNotification.offsetHeight; 
        toastNotification.classList.add('show');
        
        setTimeout(() => {
            toastNotification.classList.remove('show');
            setTimeout(() => {
                toastNotification.style.display = 'none';
            }, 300);
        }, 3000);
    }

    async function checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.ollama === 'available') {
                ollamaStatus.classList.add('available');
                ollamaStatus.title = "Ollama Fallback Available";
            } else {
                ollamaStatus.classList.remove('available');
                ollamaStatus.title = "Ollama Fallback Unavailable";
            }
        } catch (error) {
            console.error("Health check failed:", error);
        }
    }

    function confirmDeleteDocument(docName) {
        documentToDelete = docName;
        deleteModalText.textContent = `Are you sure you want to delete "${docName}"?`;
        deleteModal.style.display = 'flex';
    }

    async function loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            const docs = await response.json();
            
            docList.innerHTML = '';
            if (docs.length === 0) {
                docList.innerHTML = '<li style="color: var(--text-muted)">No documents loaded</li>';
                return;
            }
            
            docs.forEach(doc => {
                const li = document.createElement('li');
                
                const infoDiv = document.createElement('div');
                infoDiv.className = 'doc-info';
                infoDiv.innerHTML = `<span class="doc-name" title="${doc.name}">${doc.name}</span>`;
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-btn';
                deleteBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                `;
                deleteBtn.title = 'Delete Document';
                deleteBtn.onclick = () => confirmDeleteDocument(doc.name);
                
                li.appendChild(infoDiv);
                li.appendChild(deleteBtn);
                docList.appendChild(li);
            });
        } catch (error) {
            console.error('Failed to load documents:', error);
            docList.innerHTML = '<li style="color: var(--conf-none)">Failed to load documents</li>';
        }
    }

    function addMessage(text, role, metadata = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        
        // Setup Avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        if (role === 'user') {
            avatarDiv.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>';
        } else {
            avatarDiv.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"></path><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>';
        }
        
        // Setup Content Container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Badges (Confidence & Answer Source)
        if (role === 'assistant' && metadata) {
            const badgesDiv = document.createElement('div');
            badgesDiv.style.display = 'flex';
            badgesDiv.style.gap = '0.5rem';
            badgesDiv.style.alignItems = 'center';
            badgesDiv.style.marginBottom = '0.75rem';

            if (metadata.confidence_level && metadata.confidence_level !== 'none') {
                const confDiv = document.createElement('div');
                confDiv.className = `confidence-badge conf-${metadata.confidence_level}`;
                confDiv.style.marginBottom = '0';
                confDiv.innerHTML = `
                    <div class="conf-dot"></div>
                    ${metadata.confidence_level} Confidence (${(metadata.confidence * 100).toFixed(0)}%)
                `;
                badgesDiv.appendChild(confDiv);
            }
            
            // Generated By Badge
            if (metadata.answer_source) {
                const sourceBadge = document.createElement('div');
                sourceBadge.className = `source-badge source-${metadata.answer_source}`;
                
                let icon = '';
                let label = '';
                if (metadata.answer_source === 'gemini') {
                    icon = '✓'; label = 'Gemini';
                } else if (metadata.answer_source === 'ollama') {
                    icon = '✓'; label = 'Ollama Fallback';
                } else if (metadata.answer_source === 'retrieval') {
                    icon = '✓'; label = 'Retrieval Mode';
                } else if (metadata.answer_source === 'guardrails') {
                    icon = '✗'; label = 'Guardrail Refusal';
                }
                
                sourceBadge.innerHTML = `<span>${icon}</span> ${label}`;
                badgesDiv.appendChild(sourceBadge);
            }
            
            if (badgesDiv.children.length > 0) {
                contentDiv.appendChild(badgesDiv);
            }
        }
        
        // Text Bubble
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        if (role === 'assistant') {
            // Parse Markdown for assistant responses
            bubbleDiv.innerHTML = marked.parse(text);
        } else {
            bubbleDiv.textContent = text;
        }
        contentDiv.appendChild(bubbleDiv);
        
        // Sources (if any)
        if (role === 'assistant' && metadata && metadata.sources && metadata.sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.className = 'sources-title';
            sourcesTitle.textContent = 'Sources Used';
            sourcesContainer.appendChild(sourcesTitle);
            
            metadata.sources.forEach(source => {
                const clone = sourceTemplate.content.cloneNode(true);
                clone.querySelector('.source-doc').textContent = source.document;
                clone.querySelector('.source-page').textContent = `Page ${source.page}`;
                clone.querySelector('.source-excerpt').textContent = source.excerpt;
                sourcesContainer.appendChild(clone);
            });
            
            contentDiv.appendChild(sourcesContainer);
        }
        
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        
        // Scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'message assistant-message';
        
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"></path></svg>
            </div>
            <div class="message-content">
                <div class="message-bubble typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
});
