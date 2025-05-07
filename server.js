const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 5000;

// Enable CORS for all routes
app.use(cors());

// Set up multer for file upload
const upload = multer({
  dest: 'uploads/', // temporary storage
  limits: { fileSize: 5 * 1024 * 1024 }, // 5 MB limit
});

// POST route to handle audio file and run prediction
app.post('/predict', upload.single('audio'), (req, res) => {
  const audioPath = path.join(__dirname, req.file.path);

  const pythonProcess = spawn('python', ['predict.py', audioPath]);

  let result = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  pythonProcess.on('close', (code) => {
    // Clean up uploaded file
    fs.unlink(audioPath, () => {});

    if (code !== 0) {
      console.error(`Python script failed: ${errorOutput}`);
      return res.status(500).json({ error: 'Python script failed' });
    }

    try {
      const emotion = result.trim();
      return res.json({ emotion });
    } catch (e) {
      console.error('Failed to parse Python output:', e);
      return res.status(500).json({ error: 'Invalid Python output' });
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
