import React from 'react';
import AudioRecorderContainer from "../components/AudioRecorderContainer";

function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 p-6">
      <header className="text-center text-white mb-8">
        <h1 className="text-4xl font-bold drop-shadow-lg">🎧 Audio Chatbot</h1>
        <p className="text-white/80 mt-2">Record your voice and chat with the bot</p>
      </header>

      <main className="max-w-3xl mx-auto">
        <div className="bg-white rounded-2xl p-6 shadow-xl">
          <AudioRecorderContainer />
        </div>
      </main>
    </div>
  );
}

export default HomePage;
