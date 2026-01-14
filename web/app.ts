const resultsList = document.getElementById("results") as HTMLUListElement;
const predictButton = document.getElementById("predictButton") as HTMLButtonElement;
const seedButton = document.getElementById("seedButton") as HTMLButtonElement;

type IntentScore = { intent: string; score: number };

function renderResults(results: IntentScore[]): void {
  resultsList.innerHTML = "";
  if (!results.length) {
    resultsList.innerHTML = "<li>No scores yet.</li>";
    return;
  }
  results.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.intent}: ${item.score.toFixed(3)}`;
    resultsList.appendChild(li);
  });
}

predictButton.addEventListener("click", () => {
  const text = (document.getElementById("textInput") as HTMLTextAreaElement).value;
  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
    .then((res) => res.json())
    .then((data) => renderResults(data.results || []));
});

seedButton.addEventListener("click", () => {
  fetch("/api/seed", { method: "POST" })
    .then((res) => res.json())
    .then(() => alert("Sample intents loaded."));
});
