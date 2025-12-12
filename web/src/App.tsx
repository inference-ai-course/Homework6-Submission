import React, { useEffect, useState } from 'react';
import HomePage from "./pages/HomePage";
import { requestMicrophonePermission } from "./helpers/audio";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import {
  Box,
  Container,
  CssBaseline,
  Paper,
  Typography,
  Button
} from "@mui/material";

const theme = createTheme({
  palette: {
    primary: { main: '#6366f1' }, // indigo-500
    secondary: { main: '#9333ea' }, // purple-600
    background: {
      default: '#eef2ff'
    }
  },
  shape: { borderRadius: 12 }
});

function App() {
  const [hasPermission, setHasPermission] = useState<boolean>(false);

  useEffect(() => {
    requestMicrophonePermission().then((success) => {
      setHasPermission(success);
    });
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {hasPermission ? (
        <HomePage />
      ) : (
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            px: 2,
            background: 'linear-gradient(135deg, #6366f1 0%, #9333ea 100%)'
          }}
        >
          <Container maxWidth="sm">
            <Paper elevation={8} sx={{ p: 4, textAlign: 'center', borderRadius: 4 }}>
              <Typography variant="h5" component="h1" fontWeight={800} gutterBottom>
                Microphone permission required
              </Typography>
              <Typography color="text.secondary" sx={{ mb: 3 }}>
                To use the audio chatbot, please grant access to your microphone.
              </Typography>
              <Button
                variant="contained"
                color="primary"
                size="large"
                onClick={async () => setHasPermission(await requestMicrophonePermission())}
                sx={{
                  px: 3,
                  py: 1.25,
                  fontWeight: 700,
                  textTransform: 'none',
                  borderRadius: 2,
                  boxShadow: 3
                }}
              >
                Grant Permission
              </Button>
            </Paper>
          </Container>
        </Box>
      )}
    </ThemeProvider>
  );
}

export default App;
