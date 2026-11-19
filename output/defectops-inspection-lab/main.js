import { icon, logo } from './graphics.js';

const live = new URLSearchParams(location.search).get('mode') === 'live';
const navLinks = document.querySelectorAll('.global-nav a');
const target = document.querySelector(live ? '#live-comparison' : '#history');
if (live) {
  document.body.classList.add('live-mode');
  document.querySelector('#demo-mode-link').removeAttribute('aria-current');
  document.querySelector('#live-mode-link').setAttribute('aria-current', 'page');
  navLinks[1].href = '#live-comparison';
  navLinks[1].textContent = '결과 비교';
}
function updateNavigation() {
  const atTarget = !target.hidden && window.scrollY > 65 &&
    (target.getBoundingClientRect().top < innerHeight * .45 ||
      window.scrollY + innerHeight >= document.documentElement.scrollHeight - 2);
  navLinks.forEach((link, i) => link.classList.toggle('nav-current', i === Number(atTarget)));
}
window.addEventListener('scroll', updateNavigation, { passive: true });
window.addEventListener('resize', updateNavigation);
updateNavigation();
document.querySelector('#brand-icon').innerHTML = logo;
document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });
document.querySelector('#about-open').onclick = () => document.querySelector('#about').showModal();
document.querySelectorAll('[data-close]').forEach(b => { b.onclick = () => b.closest('dialog').close(); });
document.querySelectorAll('dialog').forEach(d => d.addEventListener('click', e => {
  if (e.target === d) d.close();
}));
if (live) await import('./live.js');
else await import('./app.js');
