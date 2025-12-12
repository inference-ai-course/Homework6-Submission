import React, { useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import { getMediaRecorder, playAudioBlob } from "../helpers/audio";
import { postChat } from "../api/chat";

const AudioRecorderContainer: React.FC = () => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const mediaRecorderRef = useRef<MediaRecorder>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setIsLoading(true);
    const response = await postChat(audioBlob);
    await playAudioBlob(response);
    setIsLoading(false);
    setIsRecording(false);
  };

  const handleStart = async (): Promise<void> => {
    chunksRef.current = [];
    mediaRecorderRef.current = await getMediaRecorder(chunksRef, handleRecordingComplete);
    setIsRecording(true);
  };

  const handleStop = async (): Promise<void> => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <AudioRecorder
      isRecording={isRecording}
      isLoading={isLoading}
      onStart={handleStart}
      onStop={handleStop}
    />
  );
};

export default AudioRecorderContainer;
