// Values inside sentences that you drag sideways to change, after Bret Victor's Tangle.
// Works with a mouse, a finger (the element sets touch-action: none) and the arrow keys.

export function adjustable(node, { options = null, min = 0, max = 1, step = 1, pxPerStep = 12,
  get, set, format = String, label = "" }) {
  const count = options ? options.length : Math.round((max - min) / step) + 1;
  const indexOf = () => (options ? Math.max(0, options.indexOf(get())) : Math.round((get() - min) / step));
  const valueAt = (i) => (options ? options[i] : +(min + i * step).toFixed(6));
  node.classList.add("adjust");
  node.tabIndex = 0;
  node.setAttribute("role", "slider");
  node.setAttribute("aria-label", label);
  node.setAttribute("aria-valuemin", 0);
  node.setAttribute("aria-valuemax", count - 1);

  const render = () => {
    node.textContent = format(get());
    node.setAttribute("aria-valuenow", indexOf());
    node.setAttribute("aria-valuetext", node.textContent);
  };
  const go = (i) => {
    i = Math.max(0, Math.min(count - 1, i));
    if (i !== indexOf()) set(valueAt(i));
    render();
  };

  let startX = 0, startIndex = 0, moved = false, pointer = null;
  node.addEventListener("pointerdown", (e) => {
    pointer = e.pointerId;
    startX = e.clientX;
    startIndex = indexOf();
    moved = false;
    node.setPointerCapture(e.pointerId);
    node.classList.add("dragging");
    e.preventDefault();
  });
  node.addEventListener("pointermove", (e) => {
    if (e.pointerId !== pointer) return;
    const delta = Math.round((e.clientX - startX) / pxPerStep);
    if (delta !== 0) moved = true;
    go(startIndex + delta);
  });
  const end = (e) => {
    if (e.pointerId !== pointer) return;
    pointer = null;
    node.classList.remove("dragging");
    if (!moved) go(indexOf() + 1 >= count ? 0 : indexOf() + 1); // a tap steps forward
  };
  node.addEventListener("pointerup", end);
  node.addEventListener("pointercancel", (e) => { moved = true; end(e); });
  node.addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (d) { go(indexOf() + d); e.preventDefault(); }
  });
  render();
  return render;
}
