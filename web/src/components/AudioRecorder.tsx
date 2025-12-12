import React from 'react';
import { Button } from '@mui/material';

type AudioRecorderDisplayProps = {
  isRecording: boolean,
  onStart: () => void,
  onStop: () => void,
  isLoading: boolean
};

const AudioRecorder: React.FC<AudioRecorderDisplayProps> = ({
                                                              isRecording,
                                                              onStart,
                                                              onStop,
                                                              isLoading
                                                            }) => {
  return (
    <div className="flex items-center justify-center">
      <div className="w-full flex justify-center">
        {isRecording ? (
          <Button
            variant="contained"
            color="success"
            size="large"
            onClick={onStop}
            disabled={isLoading}
            sx={{
              px: 3,
              py: 1.25,
              fontWeight: 700,
              textTransform: 'none',
              borderRadius: 2,
              boxShadow: 3
            }}
          >
            Send to Chatbot
          </Button>
        ) : (
          <Button
            variant="contained"
            color="primary"
            size="large"
            onClick={onStart}
            sx={{
              px: 3,
              py: 1.25,
              fontWeight: 700,
              textTransform: 'none',
              borderRadius: 2,
              boxShadow: 3
            }}
          >
            Start Recording
          </Button>
        )}
      </div>
    </div>
  );
};

export default AudioRecorder;
