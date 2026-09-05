/**
 * theme.js -- Dark/Light mode toggle for BAS Experiment Monitor.
 * Persists choice in localStorage so it survives app restarts.
 */
(function () {
    const STORAGE_KEY = 'bas-monitor-theme';
    const root = document.documentElement;
    const themeBtn = document.getElementById('btn-theme-toggle');
    const themeIcon = themeBtn ? themeBtn.querySelector('.material-symbols-outlined') : null;

    function applyTheme(theme) {
        if (theme === 'light') {
            root.classList.remove('dark');
            root.classList.add('light');
            if (themeIcon) themeIcon.textContent = 'light_mode';
        } else {
            root.classList.remove('light');
            root.classList.add('dark');
            if (themeIcon) themeIcon.textContent = 'dark_mode';
        }
    }

    const saved = localStorage.getItem(STORAGE_KEY) || 'dark';
    applyTheme(saved);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = root.classList.contains('light') ? 'light' : 'dark';
            const next = current === 'light' ? 'dark' : 'light';
            localStorage.setItem(STORAGE_KEY, next);
            applyTheme(next);
        });
    }
})();
