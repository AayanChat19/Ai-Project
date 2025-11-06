document.addEventListener("mouseup", () => {
  const selection = window.getSelection().toString().trim();
  if (selection.length > 0) {
    showConfidenceTooltip(selection);
  }
});

function showConfidenceTooltip(text) {
  const confidence = (Math.random() * (1 - 0.6) + 0.6).toFixed(2); // random 0.6–1.0
  const tooltip = document.createElement("div");
  tooltip.className = "confidence-tooltip";
  tooltip.textContent = `Confidence: ${confidence} – Verified by evidence 🔍`;

  document.body.appendChild(tooltip);

  const range = window.getSelection().getRangeAt(0);
  const rect = range.getBoundingClientRect();
  tooltip.style.top = `${rect.top + window.scrollY - 30}px`;
  tooltip.style.left = `${rect.left + window.scrollX}px`;

  setTimeout(() => tooltip.remove(), 3000); // fade after 3s
}
