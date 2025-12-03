// static/js/app.js

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isMuted = false;
const API_BASE_URL = 'http://localhost:8000';

// WAV encoder function to convert raw audio data to WAV format
function encodeWAV(samples, sampleRate = 16000) {
    const numChannels = 1;
    const length = samples.length * numChannels * 2 + 36;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length, true);
    
    // Write samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        view.setInt16(offset, samples[i] * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

// Convert audio blob to WAV format
async function blobToWAV(blob) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Get audio data
    const samples = audioBuffer.getChannelData(0);
    
    return encodeWAV(Array.from(samples), audioBuffer.sampleRate);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkBrowserSupport();
    setupEventListeners();
});

function checkBrowserSupport() {
    const getUserMedia = navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
    if (!getUserMedia) {
        showError('Your browser does not support audio recording. Please use a modern browser.');
        document.getElementById('recordButton').disabled = true;
    }
}

function setupEventListeners() {
    // Handle Enter key to send
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isRecording) {
            if (!document.getElementById('sendButton').disabled) {
                sendAudio();
            }
        }
    });
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            // Optionally process when recording stops
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        const recordButton = document.getElementById('recordButton');
        recordButton.classList.add('recording');
        document.getElementById('recordButtonText').textContent = 'Stop Recording';
        document.getElementById('statusText').textContent = 'Recording...';
        document.getElementById('recordingIndicator').classList.add('active');
        document.getElementById('sendButton').disabled = false;
        
    } catch (error) {
        showError(`Failed to start recording: ${error.message}`);
        isRecording = false;
    }
}

function stopRecording() {
    if (!mediaRecorder) return;
    
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    
    // Update UI
    const recordButton = document.getElementById('recordButton');
    recordButton.classList.remove('recording');
    document.getElementById('recordButtonText').textContent = 'Start Recording';
    document.getElementById('statusText').textContent = 'Ready';
    document.getElementById('recordingIndicator').classList.remove('active');
}

async function sendAudio() {
    if (audioChunks.length === 0) {
        showError('No audio recorded. Please record something first.');
        return;
    }

    // Create blob from chunks - use actual recorded format
    const recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });

    try {
        // Convert to proper WAV format
        const audioWAVBlob = await blobToWAV(recordedBlob);

        // Disable send button and show loader
        document.getElementById('sendButton').disabled = true;
        document.getElementById('recordButton').disabled = true;
        showLoader();

        // Send audio to backend
        const formData = new FormData();
        formData.append('audio', audioWAVBlob, 'audio.wav');

        const response = await fetch(`/chat/`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Backend error response:', errorText);
            throw new Error(errorText);
        }

        // Parse JSON response containing text and audio URL
        const responseData = await response.json();

        if (responseData.error) {
            throw new Error(responseData.error);
        }

        // Show user's transcribed text with audio player (local blob)
        addMessageWithAudio(responseData.user_text, 'user', audioWAVBlob);

        // Fetch bot audio from backend URL
        const audioUrl = responseData.audio_url.startsWith('http')
            ? responseData.audio_url
            : `${API_BASE_URL}${responseData.audio_url}`;
        const audioBlob = await fetch(audioUrl).then(res => res.blob());

        // Play audio
        await playAudioResponse(audioBlob);

        // Show bot's response text with audio player
        addMessageWithAudio(responseData.bot_response, 'bot', audioBlob);

    } catch (error) {
        showError(`Error: ${error.message}`);
        addMessage(`Error: ${error.message}`, 'bot');
    } finally {
        hideLoader();
        document.getElementById('sendButton').disabled = true;
        document.getElementById('recordButton').disabled = false;

        // Reset for next recording
        audioChunks = [];
        document.getElementById('statusText').textContent = 'Ready';
    }
}


async function playAudioResponse(audioBlob) {
    return new Promise((resolve) => {
        try {
            if (isMuted) {
                console.log('Audio is muted, skipping playback');
                resolve();
                return;
            }
            
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            // Set a timeout to resolve even if audio doesn't end naturally
            const timeout = setTimeout(() => {
                console.log('Audio timeout - resolving after 10 seconds');
                URL.revokeObjectURL(audioUrl);
                resolve();
            }, 10000); // 10 second timeout
            
            audio.onended = () => {
                console.log('Audio playback ended');
                clearTimeout(timeout);
                URL.revokeObjectURL(audioUrl);
                resolve();
            };
            
            audio.onerror = (error) => {
                console.error('Audio element error:', error);
                clearTimeout(timeout);
                URL.revokeObjectURL(audioUrl);
                resolve();
            };
            
            // Play the audio
            const playPromise = audio.play();
            if (playPromise !== undefined) {
                playPromise.then(() => {
                    console.log('Audio playing successfully');
                }).catch(err => {
                    console.error('Playback error:', err);
                    clearTimeout(timeout);
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                });
            }
        } catch (error) {
            console.error('Audio creation error:', error);
            resolve();
        }
    });
}

function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessageWithAudio(text, sender, audioBlob) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // Create audio player
    const audioUrl = URL.createObjectURL(audioBlob);
    const audioPlayer = document.createElement('audio');
    audioPlayer.controls = true;
    audioPlayer.style.width = '100%';
    audioPlayer.style.marginBottom = '10px';
    audioPlayer.src = audioUrl;
    
    // Create text content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(audioPlayer);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const chatInputArea = document.querySelector('.chat-input-area');
    chatInputArea.insertBefore(errorDiv, chatInputArea.firstChild);
    
    // Remove error after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showLoader() {
    document.getElementById('loader').classList.add('active');
}

function hideLoader() {
    document.getElementById('loader').classList.remove('active');
}

function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        document.getElementById('chatMessages').innerHTML = `
            <div class="message bot-message">
                <div class="message-content">Hello! I'm your AI assistant. Click the microphone button to start recording your message.</div>
            </div>
        `;
        audioChunks = [];
        document.getElementById('recordButton').disabled = false;
    }
}

function toggleMicrophone() {
    isMuted = !isMuted;
    const muteButton = document.getElementById('muteButton');
    if (isMuted) {
        muteButton.textContent = '🔇 Muted';
        muteButton.style.opacity = '0.6';
    } else {
        muteButton.textContent = '🔊 Unmute';
        muteButton.style.opacity = '1';
    }
}
