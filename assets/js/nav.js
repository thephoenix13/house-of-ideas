/* nav.js — mobile nav overlay */
(function () {
  const toggle = document.querySelector('[data-nav-toggle]');
  const overlay = document.querySelector('[data-mobile-nav]');
  const closeBtn = document.querySelector('[data-mobile-nav-close]');

  if (!toggle || !overlay) return;

  function open() {
    overlay.classList.add('is-open');
    overlay.setAttribute('aria-hidden', 'false');
    toggle.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
  }

  function close() {
    overlay.classList.remove('is-open');
    overlay.setAttribute('aria-hidden', 'true');
    toggle.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  toggle.addEventListener('click', open);
  if (closeBtn) closeBtn.addEventListener('click', close);
  overlay.querySelectorAll('a').forEach((a) => a.addEventListener('click', close));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.classList.contains('is-open')) close();
  });
})();
