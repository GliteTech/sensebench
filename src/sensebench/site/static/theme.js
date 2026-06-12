(function () {
  const key = "sensebench-theme";
  const root = document.documentElement;
  const saved = window.localStorage.getItem(key);
  if (saved === "dark" || saved === "light") {
    root.dataset.theme = saved;
  }
  const button = document.getElementById("theme-toggle");
  if (!button) {
    return;
  }
  button.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    window.localStorage.setItem(key, next);
  });
})();
