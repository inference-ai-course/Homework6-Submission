const fileInput = document.getElementById('audioFile');
const fileName = document.getElementById('fileName');
const submitBtn = document.getElementById('submitBtn');
const questionPlayer = document.getElementById('questionPlayer');
const answerPlayer = document.getElementById('answerPlayer');
const message = document.getElementById('message');
const historyPanel = document.getElementById('historyPanel');

let selectedFile = null;
let history = [];
const MAX = 10

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        fileName.textContent = `Selected: ${file.name}`;
        submitBtn.disabled = false;
        message.style.display = 'none';
        questionPlayer.style.display = "block"
        const fileURL = URL.createObjectURL(file);
        questionPlayer.src = fileURL;
        answerPlayer.style.display = "none"
        submitBtn.disabled = false
    }
});

submitBtn.addEventListener('click', async function() {
    if (selectedFile) {
        submitBtn.disabled = true
        message.style.display = 'block';
        message.className = "loading-spinner"
        answerPlayer.src = ''
        const formData = new FormData();
        formData.append('file', selectedFile);
        const response = await fetch('/api/voice-query', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            throw new Error(`Failed to call API! status: ${response.status}`);
        }

        const resp_json = await response.json();

        const audio_path = resp_json.audio_path
        answerPlayer.style.display = "block"
        answerPlayer.src = audio_path

        historyPanel.style.display = "block"
        historyPanel.innerHTML = '';
        history.push({'question': resp_json.user_text, 'answer': resp_json.response_text})
        if (history.length >= MAX) {
            history.splice(0, 1);
        }
        history.forEach(qa => {
            historyPanel.innerHTML += 'Q: ' + qa.question + '<br><br>A: ' + qa.answer + '<br><br>'
        });
        
        message.className = 'message';
        submitBtn.disabled = false
    }
});