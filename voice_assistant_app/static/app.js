/**
 * Voice Assistant Frontend
 * Handles audio recording, upload, and playback
 */

class VoiceAssistant {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.sessionId = null;
        
        // DOM elements
        this.recordButton = document.getElementById('recordButton');
        this.statusElement = document.getElementById('status');
        this.chatContainer = document.getElementById('chatContainer');
        this.sessionIdElement = document.getElementById('sessionId');
        this.messageCountElement = document.getElementById('messageCount');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorMessage = document.getElementById('errorMessage');
        this.clearButton = document.getElementById('clearButton');
        this.viewHistoryButton = document.getElementById('viewHistoryButton');
        
        this.init();
    }
    
    init() {
        // Check for microphone support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Your browser does not support audio recording');
            return;
        }
        
        // Bind event listeners
        this.recordButton.addEventListener('click', () => this.toggleRecording());
        this.clearButton.addEventListener('click', () => this.clearConversation());
        this.viewHistoryButton.addEventListener('click', () => this.viewHistory());
        
        console.log('Voice Assistant initialized');
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.addEventListener('dataavailable', event => {
                this.audioChunks.push(event.data);
            });
            
            this.mediaRecorder.addEventListener('stop', () => {
                this.processRecording();
            });
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            this.updateUI('recording', 'Recording... Click to stop');
            this.recordButton.classList.add('recording');
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showError('Failed to access microphone. Please grant permission.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            
            this.recordButton.classList.remove('recording');
            this.updateUI('processing', 'Processing...');
            
            console.log('Recording stopped');
        }
    }
    
    async processRecording() {
        this.loadingSpinner.style.display = 'block';
        
        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            
            // Create FormData
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            if (this.sessionId) {
                formData.append('session_id', this.sessionId);
            }
            
            // Send to API
            const response = await fetch('/api/voice-chat', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Update session ID if new
            if (!this.sessionId) {
                this.sessionId = result.session_id;
                this.sessionIdElement.textContent = this.sessionId.substring(0, 8) + '...';
            }
            
            // Update message count
            this.messageCountElement.textContent = result.conversation_length;
            
            // Add messages to chat
            this.addMessage('user', result.transcription);
            this.addMessage('assistant', result.response_text, result.audio_url);
            
            this.updateUI('idle', 'Ready to record');
            
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process your message. Please try again.');
            this.updateUI('idle', 'Ready to record');
        } finally {
            this.loadingSpinner.style.display = 'none';
        }
    }
    
    addMessage(role, text, audioUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const textP = document.createElement('p');
        textP.textContent = text;
        messageDiv.appendChild(textP);
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();
        messageDiv.appendChild(timestamp);
        
        // Add audio player for assistant messages
        if (role === 'assistant' && audioUrl) {
            const audio = document.createElement('audio');
            audio.className = 'audio-player';
            audio.controls = true;
            audio.src = audioUrl;
            audio.autoplay = true;
            messageDiv.appendChild(audio);
        }
        
        // Clear placeholder if it exists
        if (this.chatContainer.children.length === 1 && 
            this.chatContainer.firstChild.textContent.includes('Start a conversation')) {
            this.chatContainer.innerHTML = '';
        }
        
        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    updateUI(state, statusText) {
        this.statusElement.textContent = statusText;
        this.statusElement.className = `status ${state}`;
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
        
        setTimeout(() => {
            this.errorMessage.style.display = 'none';
        }, 5000);
    }
    
    async clearConversation() {
        if (!this.sessionId) {
            this.showError('No active session to clear');
            return;
        }
        
        if (!confirm('Are you sure you want to clear the conversation?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/session/${this.sessionId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error('Failed to clear session');
            }
            
            // Reset UI
            this.chatContainer.innerHTML = `
                <div style="text-align: center; color: #999; padding: 50px 0;">
                    Start a conversation by clicking the microphone button below!
                </div>
            `;
            this.sessionId = null;
            this.sessionIdElement.textContent = 'Not started';
            this.messageCountElement.textContent = '0';
            
            console.log('Conversation cleared');
            
        } catch (error) {
            console.error('Error clearing conversation:', error);
            this.showError('Failed to clear conversation');
        }
    }
    
    async viewHistory() {
        if (!this.sessionId) {
            this.showError('No active session');
            return;
        }
        
        try {
            const response = await fetch(`/api/session/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error('Failed to fetch history');
            }
            
            const data = await response.json();
            
            // Create modal or console log for history
            console.log('Conversation History:', data.conversation_history);
            
            // Simple alert for now (can be replaced with a modal)
            const historyText = data.conversation_history
                .map(msg => `${msg.role.toUpperCase()}: ${msg.content}`)
                .join('\n\n');
            
            alert(`Conversation History (${data.message_count} messages):\n\n${historyText}`);
            
        } catch (error) {
            console.error('Error fetching history:', error);
            this.showError('Failed to fetch conversation history');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VoiceAssistant();
});
