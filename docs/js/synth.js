// Web Audio playback. A few decaying partials per note, loosely after the additive
// piano in src/musegen/audio.py, so it needs no samples or SoundFont.

const PARTIALS = [[1, 1], [0.5, 1.5], [0.28, 2], [0.15, 2.6]];

export class Player {
  constructor() {
    this.ctx = null;
    this.voices = [];
    this.startedAt = 0;
    this.length = 0;
    this.onEnd = null;
    this.timer = null;
  }

  get playing() {
    return this.timer !== null;
  }

  // Current position in steps, or null when stopped.
  position(secondsPerStep) {
    if (!this.playing) return null;
    return (this.ctx.currentTime - this.startedAt) / secondsPerStep;
  }

  play(notes, tempo, totalSteps) {
    this.stop();
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ctx = this.ctx;
    if (ctx.state === "suspended") ctx.resume();
    const sps = 60 / tempo / 4;
    const master = ctx.createGain();
    master.gain.value = 0.18;
    master.connect(ctx.destination);
    this.master = master;
    const t0 = ctx.currentTime + 0.08;
    this.startedAt = t0;
    for (const n of notes) {
      const freq = 440 * 2 ** ((n.pitch - 69) / 12);
      const start = t0 + n.start * sps;
      const end = start + Math.max(n.duration * sps, 0.05);
      const amp = (n.velocity / 127) ** 1.5;
      const decay = 0.6 + 2.2 * (freq / 1000);
      PARTIALS.forEach(([rel, rate], k) => {
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.frequency.value = freq * (k + 1);
        g.gain.setValueAtTime(0, start);
        g.gain.linearRampToValueAtTime(amp * rel, start + 0.005);
        g.gain.setTargetAtTime(0, start + 0.005, 1 / (decay * rate));
        g.gain.setTargetAtTime(0, end, 0.03);
        osc.connect(g).connect(master);
        osc.start(start);
        osc.stop(end + 0.25);
        this.voices.push(osc);
      });
    }
    this.length = totalSteps * sps;
    this.timer = setTimeout(() => this.stop(), (this.length + 0.4) * 1000);
  }

  stop() {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    for (const v of this.voices) {
      try { v.stop(); } catch { /* already stopped */ }
    }
    this.voices = [];
    if (this.master) this.master.disconnect();
    this.master = null;
    if (this.onEnd) this.onEnd();
  }
}
