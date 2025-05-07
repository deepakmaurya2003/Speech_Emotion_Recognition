import React from 'react';
import Footer from '../components/Footer';
import emotionBg from '../emotion.png';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <div
        className="relative text-white py-20"
        style={{
          backgroundImage: `url(${emotionBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        }}
      >
        <div className="absolute inset-0 bg-indigo-700 opacity-75"></div>

        <div className="relative max-w-6xl mx-auto px-6 text-center z-10">
          <h1 className="text-white text-5xl font-bold mb-6">Speech Emotion Recognition</h1>
          <p className="text-xl text-indigo-100 max-w-3xl mx-auto">
            Mental health monitoring through voice analysis
          </p>
          <div className="mt-8">
          <a href="/record">
            <button className="bg-white text-indigo-700 px-8 py-3 rounded-lg font-semibold hover:bg-indigo-100 transition duration-300 mr-4">
              Try Demo
            </button>
            </a>
            <a href="/about">
            <button className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-indigo-700 transition duration-300">
              Learn More
            </button>
            </a>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-16 flex-grow">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-indigo-700 mb-4">Key Features</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            "A real-time monitoring system for emotion detection, enabling longitudinal analysis to better understand behavioral trends and support the emotional well-being of patients in healthcare."          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 mb-16">
          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition duration-300">
            <div className="bg-indigo-100 ml-[110px] w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-indigo-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="ml-12 text-xl font-semibold text-indigo-600 mb-2">Real-Time Analysis</h3>
            <p className="text-gray-700">
              Continuous emotion tracking, enabling immediate feedback and interventions.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition duration-300">
            <div className="bg-indigo-100 ml-[110px] w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-indigo-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold ml-[60px] text-indigo-600 mb-2">Audio Recording</h3>
            <p className="text-gray-700">
              Allows audio collection remotely for non-intrusive monitoring.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition duration-300">
            <div className="bg-indigo-100 ml-[110px] w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-indigo-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <h3 className="text-xl ml-[60px] font-semibold text-indigo-600 mb-2">Interactive Interface</h3>
            <p className="text-gray-700">
            Web-based interface for recording and monitoring emotional patterns over time.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 mb-16">
          <h2 className="text-3xl font-bold text-indigo-700 mb-8 text-center">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-indigo-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-indigo-700">1</span>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">Speak Naturally</h3>
              <p className="text-gray-700">
                Just talk as you normally would - no special phrases or acting required.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-indigo-700">2</span>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">AI Analysis</h3>
              <p className="text-gray-700">
                Our deep learning model extracts features from your voice.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-indigo-700">3</span>
              </div>
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">Get Insights</h3>
              <p className="text-gray-700">
                Receive immediate feedback about detected emotions and patterns.
              </p>
            </div>
          </div>
        </div>

      </div>

      <Footer />
    </div>
  );
}