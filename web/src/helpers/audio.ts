import { MutableRefObject } from "react";

export const requestMicrophonePermission = async (): Promise<boolean> => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Stop the stream immediately, we'll request it again when recording starts
    for (const track of stream.getTracks()) track.stop();
    return true
  } catch {
    alert('Microphone access is required to record audio. Please grant permission.');
    return false
  }
};

export const getMediaRecorder = async (chunksRef: MutableRefObject<Blob[]>, onRecordingComplete: (blob: Blob) => void): Promise<MediaRecorder> => {
    const stream: MediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44_100
      }
    })

    const mediaRecorder = new MediaRecorder(stream)
    chunksRef.current = []

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data as Blob);
      }
    }

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
      onRecordingComplete(audioBlob);
      for (const track of stream.getTracks()) track.stop();
    };

    mediaRecorder.start();
    return mediaRecorder;
}

export const playAudioBlob = async (
  blob: Blob,
): Promise<void> => {

  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);

  const onEnded = () => {
    URL.revokeObjectURL(url);
    audio.removeEventListener('ended', onEnded);
  };
  audio.addEventListener('ended', onEnded);

  await audio.play();
};