import React from 'react';
import Footer from '../components/Footer';

export default function About() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-indigo-50 to-white">
      <div className="bg-indigo-700 text-white py-16">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h1 className="text-white text-4xl md:text-5xl font-bold mb-6">About Our Speech Emotion Recognition</h1>
          <p className="text-xl text-indigo-100 max-w-3xl mx-auto">
            Mental health monitoring through voice analysis
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 flex-grow">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold text-indigo-700 mb-6">Our Mission</h2>
          <div className="bg-white rounded-xl shadow-lg p-8 max-w-4xl mx-auto">
            <p className="text-gray-700 text-lg mb-6">
              Emotional health is neglected in today's busy world. Traditional methods of mental health tracking rely on infrequent check-ins and subjective feelings, which may fail to pick up on subtle but significant emotional changes.
              We aim to revolutionize emotional monitoring through AI-powered Speech Emotion Recognition—offering real-time, objective insights into emotional states using just the human voice.This technology can support hospitals by enabling real-time emotional analysis of patients, helping mental health professionals detect early signs of distress and monitor recovery more effectively.  </p>

          </div>
        </div>

        <div className="mb-16">
          <h2 className="text-3xl font-bold text-indigo-700 mb-8 text-center">Technology Highlights</h2>
          <div className="grid gap-8 md:grid-cols-2">
            <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition duration-300">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 p-3 rounded-lg mr-4">
                  <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-indigo-600">RAVDESS Dataset</h3>
              </div>
              <p className="text-gray-700">
                Our model is trained on the Ryerson Audio-Visual Database, containing 7,356 professionally acted vocal expressions across 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) with both speech and song variations.
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition duration-300">
              <div className="flex items-center mb-4">
                <div className="bg-indigo-100 p-3 rounded-lg mr-4">
                  <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-indigo-600">CNN Architecture</h3>
              </div>
              <p className="text-gray-700">
              Our Speech Emotion Recognition system is powered by a Convolutional Neural Network (CNN) trained on spectrogram inputs derived from real-time audio signals.  Features like  Mel-Frequency Cepstral Coefficients (MFCCs) and spectrogram coefficients capture both spectral and temporal nuances of speech, allowing the CNN to effectively learn and differentiate emotional patterns.           </p>
            </div>
          </div>
        </div>

        <div className="mb-16">
          <h2 className="text-3xl font-bold text-indigo-700 mb-8 text-center">Real-World Applications</h2>
          <div className="grid gap-8 md:grid-cols-3">
            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="bg-indigo-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-3">Therapy Monitoring</h3>
              <p className="text-gray-700">
                Track patient emotional progress between sessions through voice journaling with automatic emotion tracking.
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="bg-indigo-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-3">Call Center Analytics</h3>
              <p className="text-gray-700">
                Analyze customer service interactions to identify frustrated callers and improve service quality.
              </p>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 text-center">
              <div className="bg-indigo-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-3">Education</h3>
              <p className="text-gray-700">
                Help children with autism spectrum disorder better understand emotional cues through interactive voice games.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-indigo-700 text-white rounded-xl shadow-lg p-8 ">
          <h2 className="text-3xl font-bold mb-6">Our Roadmap</h2>
          <div className="space-y-6">
            <div className="flex items-start">
              <div className="bg-white text-indigo-700 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mr-4 mt-1">
                <span className="font-bold">1</span>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Multimodal Emotion Recognition</h3>
                <p className="text-indigo-100 text-left">
                We aim to integrate additional data sources such as facial expressions and physiological signals (e.g., heart rate, skin conductance) to build a more comprehensive and accurate emotion detection system.                </p>
              </div>
            </div>

            <div className="flex items-start">
              <div className="bg-white text-indigo-700 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mr-4 mt-1">
                <span className="font-bold">2</span>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Language and Accent Adaptability</h3>
                <p className="text-indigo-100 text-left">
                Future versions will focus on enhancing performance across diverse languages, dialects, and speaking styles to ensure broader applicability and inclusivity in global healthcare environments.                </p>
              </div>
            </div>

            <div className="flex items-start">
              <div className="bg-white text-indigo-700 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mr-4 mt-1">
                <span className="font-bold">3</span>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">Mobile and Edge Deployment</h3>
                <p className="text-indigo-100 text-left">
                We plan to optimize the model for deployment on mobile devices and edge hardware, allowing offline emotion detection in remote or resource-constrained settings—especially useful for telemedicine and rural healthcare.                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}