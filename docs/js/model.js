// Port of src/musegen/models/transformer.py for inference, one token at a time.
// Pre-LayerNorm blocks, rotary position embeddings, GELU (erf form), tied output head,
// and a key/value cache trimmed to the last max_len positions.

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

export function decodeFloat16(buffer) {
  const half = new Uint16Array(buffer);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]);
  return out;
}

// Abramowitz & Stegun 7.1.26. Its error (1.5e-7) is far below the float16 weight error,
// so GELU still matches torch's exact erf form to float32 precision.
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027
    + t * 1.061405429)))) * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}

const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function layerNorm(x, w, b, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// y = W x (+ b), with W stored row-major as [out, in] like nn.Linear.
// cols is always a multiple of 4 here (d_model and 4 * d_model), so the loop is unrolled.
function linear(W, x, rows, cols, out, bias = null) {
  for (let r = 0; r < rows; r++) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    const o = r * cols;
    for (let c = 0; c < cols; c += 4) {
      s0 += W[o + c] * x[c];
      s1 += W[o + c + 1] * x[c + 1];
      s2 += W[o + c + 2] * x[c + 2];
      s3 += W[o + c + 3] * x[c + 3];
    }
    out[r] = (bias ? bias[r] : 0) + s0 + s1 + s2 + s3;
  }
  return out;
}

export class Transformer {
  constructor(manifest, weights) {
    const c = manifest.config;
    this.V = c.vocab_size;
    this.d = c.d_model;
    this.L = c.n_layers;
    this.H = c.n_heads;
    this.hd = c.d_model / c.n_heads;
    this.ff = c.ff_mult * c.d_model;
    this.maxLen = c.max_len;
    this.w = {};
    for (const t of manifest.tensors) this.w[t.name] = weights.subarray(t.offset, t.offset + t.length);
    const half = this.hd / 2;
    this.invFreq = new Float64Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = 1 / 10000 ** ((2 * i) / this.hd);
    // scratch buffers
    this.x = new Float32Array(this.d);
    this.h = new Float32Array(this.d);
    this.qkv = new Float32Array(3 * this.d);
    this.att = new Float32Array(this.d);
    this.tmp = new Float32Array(this.d);
    this.ffBuf = new Float32Array(this.ff);
    this.logits = new Float32Array(this.V);
  }

  static async load(baseUrl, fetchFn = fetch) {
    const manifest = await (await fetchFn(`${baseUrl}/model.json`)).json();
    const buf = await (await fetchFn(`${baseUrl}/model.bin`)).arrayBuffer();
    return new Transformer(manifest, decodeFloat16(buf));
  }

  // One flat [max_len + 1, d] buffer per layer for keys and for values.
  newState() {
    const size = (this.maxLen + 1) * this.d;
    return {
      pos: 0, n: 0,
      k: Array.from({ length: this.L }, () => new Float32Array(size)),
      v: Array.from({ length: this.L }, () => new Float32Array(size)),
      scores: new Float64Array(this.maxLen + 1),
    };
  }

  rotate(vec, offset, pos) {
    const half = this.hd / 2;
    for (let i = 0; i < half; i++) {
      const a = pos * this.invFreq[i];
      const cos = Math.fround(Math.cos(a)), sin = Math.fround(Math.sin(a));
      const x1 = vec[offset + i], x2 = vec[offset + i + half];
      vec[offset + i] = x1 * cos - x2 * sin;
      vec[offset + i + half] = x2 * cos + x1 * sin;
    }
  }

  // Feed one token; returns logits for the next token (a view, copy it to keep it).
  step(token, state) {
    const { d, H, hd, w } = this;
    const pos = state.pos;
    const x = this.x;
    x.set(w["embed.weight"].subarray(token * d, token * d + d));
    for (let l = 0; l < this.L; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], this.h);
      linear(w[p + "attn.qkv.weight"], this.h, 3 * d, d, this.qkv);
      const q = this.qkv.subarray(0, d);
      const k = this.qkv.subarray(d, 2 * d);
      for (let hIdx = 0; hIdx < H; hIdx++) {
        this.rotate(q, hIdx * hd, pos);
        this.rotate(k, hIdx * hd, pos);
      }
      const ks = state.k[l], vs = state.v[l];
      const n = state.n + 1; // cached positions plus this one
      ks.set(k, state.n * d);
      vs.set(this.qkv.subarray(2 * d, 3 * d), state.n * d);
      const scale = 1 / Math.sqrt(hd);
      const scores = state.scores;
      for (let hIdx = 0; hIdx < H; hIdx++) {
        const o = hIdx * hd;
        let max = -Infinity;
        for (let j = 0; j < n; j++) {
          let s = 0;
          const base = j * d + o;
          for (let i = 0; i < hd; i++) s += q[o + i] * ks[base + i];
          scores[j] = s * scale;
          if (scores[j] > max) max = scores[j];
        }
        let sum = 0;
        for (let j = 0; j < n; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
        this.att.fill(0, o, o + hd);
        for (let j = 0; j < n; j++) {
          const wj = scores[j] / sum, base = j * d + o;
          for (let i = 0; i < hd; i++) this.att[o + i] += wj * vs[base + i];
        }
      }
      // like the PyTorch model: attend over the cache plus this step, then trim the cache
      if (n > this.maxLen) {
        ks.copyWithin(0, d, n * d);
        vs.copyWithin(0, d, n * d);
      }
      linear(w[p + "attn.proj.weight"], this.att, d, d, this.tmp);
      for (let i = 0; i < d; i++) x[i] += this.tmp[i];
      layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], this.h);
      linear(w[p + "mlp.0.weight"], this.h, this.ff, d, this.ffBuf, w[p + "mlp.0.bias"]);
      for (let i = 0; i < this.ff; i++) this.ffBuf[i] = gelu(this.ffBuf[i]);
      linear(w[p + "mlp.2.weight"], this.ffBuf, d, this.ff, this.tmp, w[p + "mlp.2.bias"]);
      for (let i = 0; i < d; i++) x[i] += this.tmp[i];
    }
    layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], this.h);
    const head = w["head.weight"] || w["embed.weight"];
    linear(head, this.h, this.V, d, this.logits);
    state.pos = pos + 1;
    state.n = Math.min(state.n + 1, this.maxLen);
    return this.logits;
  }
}
