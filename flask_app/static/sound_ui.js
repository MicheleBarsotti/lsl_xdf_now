let streams = [];
const windowDuration = 30 * 1000;  // 30 seconds in ms

function getParams(idx) {
  return {
    minVal: parseFloat(document.getElementById(`minVal-${idx}`).value),
    maxVal: parseFloat(document.getElementById(`maxVal-${idx}`).value),
    minFreq: parseFloat(document.getElementById(`minFreq-${idx}`).value),
    maxFreq: parseFloat(document.getElementById(`maxFreq-${idx}`).value),
    soundOn: document.getElementById(`soundOn-${idx}`).checked
  };
}

async function poll(streamIdx, streamId) {
  const stream = streams[streamIdx];
  stream.pollActive = true;
  while (stream.pollActive) {
    try {
      const res = await fetch(`/feature/${streamId}`);
      const text = await res.text();
      const value = parseFloat(text);

      const { minVal, maxVal, minFreq, maxFreq, soundOn } = getParams(streamIdx);

      const clamped = Math.min(Math.max(value, minVal), maxVal);
      const norm = (clamped - minVal) / (maxVal - minVal || 1);
      const freq = minFreq + norm * (maxFreq - minFreq);

      if (stream.oscillator && soundOn) {
        stream.oscillator.frequency.setTargetAtTime(freq, stream.audioCtx.currentTime, 0.05);
      }

      // Keep buffer updated
      const now = Date.now();
      stream.gsrValues.push(value);
      stream.gsrTimestamps.push(now);

      while (stream.gsrTimestamps.length > 0 && now - stream.gsrTimestamps[0] > windowDuration) {
        stream.gsrValues.shift();
        stream.gsrTimestamps.shift();
      }

      drawGSR(streamIdx);
    } catch (e) {
      console.error(e);
    }
    await new Promise(r => setTimeout(r, 100));
  }
}

function updateSoundRouting(streamIdx) {
  const stream = streams[streamIdx];
  const { soundOn } = getParams(streamIdx);
  if (!stream.gainNode) return;
  try {
    stream.gainNode.disconnect();
    stream.analyser.disconnect();
  } catch (e) {}
  stream.gainNode.connect(stream.analyser);
  if (soundOn) {
    stream.analyser.connect(stream.audioCtx.destination);
  }
}

function startSound(streamIdx) {
  const stream = streams[streamIdx];
  if (stream.audioCtx) return;

  stream.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  stream.oscillator = stream.audioCtx.createOscillator();
  stream.gainNode = stream.audioCtx.createGain();
  stream.analyser = stream.audioCtx.createAnalyser();

  stream.oscillator.type = 'sine';
  const { minFreq } = getParams(streamIdx);
  stream.oscillator.frequency.setValueAtTime(minFreq, stream.audioCtx.currentTime);
  stream.gainNode.gain.setValueAtTime(0.2, stream.audioCtx.currentTime);

  stream.oscillator.connect(stream.gainNode);
  stream.gainNode.connect(stream.analyser);

  updateSoundRouting(streamIdx);

  // Listen for changes to the soundOn checkbox
  document.getElementById(`soundOn-${streamIdx}`).addEventListener('change', () => {
    updateSoundRouting(streamIdx);
  });

  stream.analyser.fftSize = 1024;
  stream.bufferLength = stream.analyser.fftSize;
  stream.dataArray = new Uint8Array(stream.bufferLength);

  stream.oscillator.start();

  poll(streamIdx, stream.streamId);
  drawOscilloscope(streamIdx);
}

function stopSound(streamIdx) {
  const stream = streams[streamIdx];
  stream.pollActive = false;
  if (stream.oscillator) {
    stream.oscillator.stop();
    stream.oscillator.disconnect();
    stream.oscillator = null;
  }
  if (stream.animationId) {
    cancelAnimationFrame(stream.animationId);
    stream.animationId = null;
  }
  if (stream.audioCtx) {
    stream.audioCtx.close();
    stream.audioCtx = null;
  }
}

function drawGSR(streamIdx) {
  const stream = streams[streamIdx];
  const canvas = document.getElementById(`gsrPlot-${stream.id}`);
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw title
  ctx.fillStyle = "white";
  ctx.font = "18px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(stream.alias, canvas.width / 2, 24);

  // Axes margins
  const marginLeft = 50;
  const marginBottom = 30;
  const plotWidth = canvas.width - marginLeft - 10;
  const plotHeight = canvas.height - marginBottom - 40;

  // Draw axes
  ctx.strokeStyle = "white";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(marginLeft, 40);
  ctx.lineTo(marginLeft, 40 + plotHeight);
  ctx.lineTo(marginLeft + plotWidth, 40 + plotHeight);
  ctx.stroke();

  if (stream.gsrValues.length < 2) return;

  // Normalize x
  const now = Date.now();
  const timeWindow = windowDuration;
  const xs = stream.gsrTimestamps.map(
    t => marginLeft + ((t - (now - timeWindow)) / timeWindow) * plotWidth
  );

  // Autoscale y
  const minY = Math.min(...stream.gsrValues);
  const maxY = Math.max(...stream.gsrValues);
  const rangeY = maxY - minY || 1;

  // y-axis ticks and labels
  ctx.fillStyle = "white";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const yVal = minY + (rangeY * (4 - i)) / 4;
    const y = 40 + (plotHeight * i) / 4;
    ctx.fillText(yVal.toFixed(2), marginLeft - 5, y + 4);
    ctx.beginPath();
    ctx.moveTo(marginLeft - 3, y);
    ctx.lineTo(marginLeft, y);
    ctx.stroke();
  }

  // x-axis ticks and labels (time in seconds)
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i++) {
    const t = (timeWindow * i) / 5;
    const x = marginLeft + (plotWidth * i) / 5;
    ctx.fillText(
      `${Math.round((t - timeWindow) / 1000)}s`,
      x,
      40 + plotHeight + 18
    );
    ctx.beginPath();
    ctx.moveTo(x, 40 + plotHeight);
    ctx.lineTo(x, 40 + plotHeight + 5);
    ctx.stroke();
  }

  // Draw GSR line
  ctx.beginPath();
  ctx.strokeStyle = "deepskyblue";
  ctx.lineWidth = 2;

  for (let i = 0; i < stream.gsrValues.length; i++) {
    const x = xs[i];
    const y =
      40 +
      ((maxY - stream.gsrValues[i]) / rangeY) * plotHeight;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawOscilloscope(streamIdx) {
  const stream = streams[streamIdx];
  const canvas = document.getElementById(`oscilloscope-${stream.id}`);
  const canvasCtx = canvas.getContext("2d");

  function drawLoop() {
    stream.animationId = requestAnimationFrame(drawLoop);
    if (!stream.analyser) return;
    stream.analyser.getByteTimeDomainData(stream.dataArray);

    canvasCtx.fillStyle = "black";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "lime";
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / stream.bufferLength;
    let x = 0;

    for (let i = 0; i < stream.bufferLength; i++) {
      const v = stream.dataArray[i] / 128.0;
      const y = v * canvas.height / 2;
      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }

  drawLoop();
}

// On page load, fetch configs and build UI
window.onload = async function() {
  const res = await fetch('/stream_configs');
  const configs = await res.json();
  const container = document.getElementById('streamsContainer');
  streams = [];

  configs.forEach((cfg, idx) => {
    // Only add row if alias is present (non-empty stream)
    if (!cfg.alias) return;

    // Add plot row for stream
    const row = document.createElement('div');
    row.className = 'plot-row';
    row.id = `plot-row-${idx}`;
    row.innerHTML = `
      <canvas id="oscilloscope-${idx}" width="200" height="200"></canvas>
      <canvas id="gsrPlot-${idx}" width="600" height="200"></canvas>
    `;
    container.appendChild(row);

    // Add controls for stream (below the row)
    const controlPanel = document.createElement('div');
    controlPanel.className = 'control-panel';
    controlPanel.innerHTML = `
      <div class="control-group">
        <label>Input Min:</label>
        <input type="number" id="minVal-${idx}" value="0.0" step="0.01">
      </div>
      <div class="control-group">
        <label>Input Max:</label>
        <input type="number" id="maxVal-${idx}" value="10" step="0.01">
      </div>
      <div class="control-group">
        <label>Min Freq (Hz):</label>
        <input type="number" id="minFreq-${idx}" value="80" step="1">
      </div>
      <div class="control-group">
        <label>Max Freq (Hz):</label>
        <input type="number" id="maxFreq-${idx}" value="2000" step="1">
      </div>
      <div class="control-group">
        <label>Sound On:</label>
        <input type="checkbox" id="soundOn-${idx}" checked>
      </div>
      <div class="control-group">
        <button onclick="startSound(${idx})">Start</button>
        <button onclick="stopSound(${idx})">Stop</button>
      </div>
    `;
    container.appendChild(controlPanel);

    // Add stream object
    streams.push({
      audioCtx: null,
      oscillator: null,
      gainNode: null,
      analyser: null,
      dataArray: null,
      bufferLength: null,
      animationId: null,
      gsrValues: [],
      gsrTimestamps: [],
      pollActive: false,
      id: idx,
      streamId: cfg.idx,
      alias: cfg.alias
    });
  });
};
