import React, { useState } from 'react';
import './Record.css';
import Footer from '../components/Footer';

const Record = () => {
  const [audioURL, setAudioURL] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [emotionResult, setEmotionResult] = useState('');

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    let chunks = [];

    recorder.ondataavailable = (event) => {
      chunks.push(event.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      setAudioBlob(blob);
      setAudioURL(url);
    };

    recorder.start();
    setMediaRecorder(recorder);
    setIsRecording(true);
    setEmotionResult('');
  };

  const stopRecording = () => {
    mediaRecorder.stop();
    setIsRecording(false);
  };

  const submitRecording = async () => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      setEmotionResult(`Detected Emotion: ${result.emotion || 'Unknown'}`);
    } catch (error) {
      console.error(error);
      setEmotionResult('Error in prediction.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-indigo-50 to-white">
      <div className="flex-grow p-6 max-w-4xl mx-auto w-full">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="text-center mb-10">
            <h1 className="text-4xl font-bold text-indigo-700 mb-3">
              <span className="inline-block mr-2">🎤</span>
              Speech Emotion Recognition
            </h1>
            <p className="text-lg text-gray-600 max-w-lg mx-auto">
              Record your voice to analyze emotional tone using AI technology
            </p>
          </div>

          {audioURL && (
            <div className="mb-8 bg-indigo-50 rounded-xl p-4">
              <h3 className="text-sm font-medium text-indigo-600 mb-2">YOUR RECORDING</h3>
              <audio 
                controls 
                src={audioURL} 
                className="w-full rounded-lg"
              />
            </div>
          )}

          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <button
              onClick={startRecording}
              disabled={isRecording}
              className={`flex items-center px-6 py-3 rounded-xl font-semibold transition-colors ${
                isRecording
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md'
              }`}
            >
              <svg 
                className={`w-5 h-5 mr-2 ${isRecording ? 'text-gray-500' : 'text-white'}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              {isRecording ? 'Recording...' : 'Start Recording'}
            </button>

            <button
              onClick={stopRecording}
              disabled={!isRecording}
              className={`flex items-center px-6 py-3 rounded-xl font-semibold transition-colors ${
                !isRecording
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-red-600 text-white hover:bg-red-700 shadow-md'
              }`}
            >
              <svg 
                className={`w-5 h-5 mr-2 ${!isRecording ? 'text-gray-500' : 'text-white'}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
              Stop Recording
            </button>

            <button
              onClick={submitRecording}
              disabled={!audioBlob}
              className={`flex items-center px-6 py-3 rounded-xl font-semibold transition-colors ${
                !audioBlob
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-700 shadow-md'
              }`}
            >
              <svg 
                className={`w-5 h-5 mr-2 ${!audioBlob ? 'text-gray-500' : 'text-white'}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Analyze Emotion
            </button>
          </div>

          {emotionResult && (
            <div className={`p-4 rounded-lg text-center text-lg font-medium ${
              emotionResult.includes('Error') 
                ? 'bg-red-100 text-red-800' 
                : 'bg-indigo-100 text-indigo-800'
            }`}>
              {emotionResult}
            </div>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Record;