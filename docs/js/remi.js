// Port of src/musegen/data/tokenizer.py: REMI tokens and the grammar used to mask sampling.

export const PAD = 0, BOS = 1, EOS = 2, BAR = 3, POS = 4, VEL = 5, PITCH = 6, DUR = 7;
export const TYPE_NAMES = ["PAD", "BOS", "EOS", "BAR", "POS", "VEL", "PITCH", "DUR"];

export class REMITokenizer {
  constructor({ steps_per_bar = 16, min_pitch = 21, max_pitch = 108, max_duration = 32,
    velocity_bins = 8 } = {}) {
    this.stepsPerBar = steps_per_bar;
    this.minPitch = min_pitch;
    this.maxPitch = max_pitch;
    this.maxDuration = max_duration;
    this.velocityBins = velocity_bins;
    this.bosId = 1; this.eosId = 2; this.barId = 3;
    this.posOffset = 4;
    this.velOffset = this.posOffset + steps_per_bar;
    this.pitchOffset = this.velOffset + velocity_bins;
    this.nPitches = max_pitch - min_pitch + 1;
    this.durOffset = this.pitchOffset + this.nPitches;
    this.vocabSize = this.durOffset + max_duration;
    this.types = new Uint8Array(this.vocabSize);
    this.types.set([PAD, BOS, EOS, BAR], 0);
    this.types.fill(POS, this.posOffset, this.velOffset);
    this.types.fill(VEL, this.velOffset, this.pitchOffset);
    this.types.fill(PITCH, this.pitchOffset, this.durOffset);
    this.types.fill(DUR, this.durOffset);
  }

  velToken(velocity) {
    const bin = Math.min(this.velocityBins - 1, Math.floor(((velocity | 0) - 1) * this.velocityBins / 127));
    return this.velOffset + Math.max(0, bin);
  }
  velocityOf(token) {
    // Python's round() rounds half to even; no bin centre lands on .5 for 8 bins.
    return Math.round((token - this.velOffset + 0.5) * 127 / this.velocityBins);
  }
  pitchOf(token) { return token - this.pitchOffset + this.minPitch; }

  describe(token) {
    const t = this.types[token];
    if (t === POS) return `POS_${token - this.posOffset}`;
    if (t === VEL) return `VEL_${token - this.velOffset}`;
    if (t === PITCH) return `PITCH_${this.pitchOf(token)}`;
    if (t === DUR) return `DUR_${token - this.durOffset + 1}`;
    return TYPE_NAMES[t];
  }

  encode(notes, nBars = 0, addBos = true, addEos = true) {
    const sorted = notes
      .filter((n) => n.pitch >= this.minPitch && n.pitch <= this.maxPitch)
      .slice()
      .sort((a, b) => a.start - b.start || a.pitch - b.pitch || a.duration - b.duration);
    const lastBar = sorted.length ? Math.floor(sorted[sorted.length - 1].start / this.stepsPerBar) : -1;
    nBars = Math.max(nBars, lastBar + 1);
    const out = addBos ? [this.bosId] : [];
    let i = 0;
    for (let bar = 0; bar < nBars; bar++) {
      out.push(this.barId);
      const barEnd = (bar + 1) * this.stepsPerBar;
      let current = null;
      while (i < sorted.length && sorted[i].start < barEnd) {
        const n = sorted[i];
        const pos = n.start - bar * this.stepsPerBar;
        if (pos !== current) { out.push(this.posOffset + pos); current = pos; }
        const dur = Math.min(Math.max(n.duration, 1), this.maxDuration);
        out.push(this.velToken(n.velocity), this.pitchOffset + n.pitch - this.minPitch,
          this.durOffset + dur - 1);
        i++;
      }
    }
    if (addEos) out.push(this.eosId);
    return out;
  }

  // Ungrammatical fragments are skipped, never thrown on (same as the Python decoder).
  decode(tokens) {
    const notes = [];
    let bar = -1, pos = 0, velocity = 80, pitch = null;
    for (const tok of tokens) {
      if (tok < 0 || tok >= this.vocabSize) continue;
      const t = this.types[tok];
      if (t === EOS) break;
      if (t === BAR) { bar += 1; pos = 0; pitch = null; }
      else if (t === POS) { pos = tok - this.posOffset; pitch = null; }
      else if (t === VEL) velocity = this.velocityOf(tok);
      else if (t === PITCH) pitch = this.pitchOf(tok);
      else if (t === DUR && pitch !== null) {
        notes.push({ start: Math.max(bar, 0) * this.stepsPerBar + pos, pitch,
          duration: tok - this.durOffset + 1, velocity });
        pitch = null;
      }
    }
    return notes;
  }
}

// Where a partial sequence is in the grammar, and which tokens may come next.
//   BOS -> BAR;  BAR -> BAR | POS | EOS;  POS -> VEL -> PITCH -> DUR;
//   DUR -> VEL (polyphonic only) | POS (later in the bar) | BAR | EOS
export class GrammarState {
  constructor(tok, { key = null, monophonic = true } = {}) {
    this.tok = tok;
    this.monophonic = monophonic;
    this.lastType = null;
    this.position = -1;
    this.bars = 0;
    this.pitchMask = new Uint8Array(tok.vocabSize);
    for (let p = tok.minPitch; p <= tok.maxPitch; p++) {
      if (!key || key.contains(p)) this.pitchMask[tok.pitchOffset + p - tok.minPitch] = 1;
    }
  }

  update(token) {
    const t = this.tok.types[token];
    if (t === BAR) { this.bars += 1; this.position = -1; }
    else if (t === POS) this.position = token - this.tok.posOffset;
    this.lastType = t;
  }

  feed(tokens) {
    for (const t of tokens) this.update(t);
    return this;
  }

  allowed(allowEos = true) {
    const { tok } = this;
    const types = tok.types;
    const mask = new Uint8Array(tok.vocabSize);
    const last = this.lastType;
    const byType = (ty) => { for (let i = 0; i < mask.length; i++) if (types[i] === ty) mask[i] = 1; };
    if (last === null) byType(BOS);
    else if (last === BOS) byType(BAR);
    else if (last === POS) byType(VEL);
    else if (last === VEL) mask.set(this.pitchMask);
    else if (last === PITCH) byType(DUR);
    else if (last === BAR || last === DUR) {
      byType(BAR);
      mask.fill(1, tok.posOffset + this.position + 1, tok.velOffset);
      if (last === DUR && !this.monophonic) byType(VEL);
    } else byType(PAD);
    if (allowEos && (last === BAR || last === DUR)) mask[tok.eosId] = 1;
    return mask;
  }
}
