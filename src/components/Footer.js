import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-indigo-700 text-white py-8 mt-12">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4 ml-8">Speech Emotion Recognition</h3>
            <p className="text-indigo-100">
              Using machine learning to analyze speech patterns and identify emotions, supporting better mental health care.           </p>
          </div>

          <div className='ml-20'>
            <h3 className="text-xl font-bold mb-4 ">Quick Links</h3>
            <ul className="space-y-2">
              <li><a href="/" className="text-indigo-100 hover:text-white">Home</a></li>
              <li><a href="/record" className="text-indigo-100 hover:text-white">Record</a></li>
              <li><a href="/about" className="text-indigo-100 hover:text-white">About</a></li>
            </ul>
          </div>

          <div>
            <h3 className=" ml-20 text-xl font-bold mb-4">Contact Us</h3>
            <p className=" mr-20 text-indigo-100">Email: abc.123@gmail.com</p>
            <p className="mr-20 text-indigo-100">Phone: (+91) 123456789</p>
          </div>
        </div>

        <div className="border-t border-indigo-500 mt-8 pt-6 text-center text-indigo-100">
          <p>&copy; {new Date().getFullYear()} Speech Emotion Recognition. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;