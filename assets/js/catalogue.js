(function () {
  'use strict';

  var DOMAIN_COLORS = {
    'AI & Intelligent Systems':           '#4A7FA5',
    'EdTech & Learning Infrastructure':   '#4A8F7E',
    'HealthTech & BioTech':               '#A0524A',
    'FinTech & Financial Infrastructure': '#6B5B9A',
    'AgriTech & Rural Infrastructure':    '#5C7A4A',
    'Climate Tech & Sustainability':      '#3A6B5F',
    'Enterprise SaaS & B2B Tools':        '#B8862F',
    'GovTech & Civic Infrastructure':     '#2B4A7A',
    'Legal Tech & Compliance':            '#7A3A4A',
    'Consumer & Community Platforms':     '#A05A3A'
  };

  var DOMAIN_LABELS = {
    'AI & Intelligent Systems':           'AI & Systems',
    'EdTech & Learning Infrastructure':   'EdTech',
    'HealthTech & BioTech':               'HealthTech',
    'FinTech & Financial Infrastructure': 'FinTech',
    'AgriTech & Rural Infrastructure':    'AgriTech',
    'Climate Tech & Sustainability':      'Climate Tech',
    'Enterprise SaaS & B2B Tools':        'Enterprise SaaS',
    'GovTech & Civic Infrastructure':     'GovTech',
    'Legal Tech & Compliance':            'Legal Tech',
    'Consumer & Community Platforms':     'Consumer'
  };

  var grid    = document.getElementById('catalogue-grid');
  var countEl = document.getElementById('catalogue-count');
  var emptyEl = document.getElementById('catalogue-empty');
  var pillBar = document.getElementById('domain-pills');

  var allIdeas = [];

  function escapeHtml(str) {
    var d = document.createElement('div');
    d.appendChild(document.createTextNode(str));
    return d.innerHTML;
  }

  function buildScoreBar(score, color) {
    var segs = '';
    for (var i = 0; i < 10; i++) {
      if (i < score) {
        segs += '<span class="dossier-card__score-seg is-filled"></span>';
      } else {
        segs += '<span class="dossier-card__score-seg"></span>';
      }
    }
    return segs;
  }

  function renderCards(ideas) {
    grid.innerHTML = '';
    var label = ideas.length === 1 ? '1 idea' : ideas.length + ' ideas';
    countEl.textContent = label;

    if (ideas.length === 0) {
      emptyEl.hidden = false;
      grid.hidden = true;
      return;
    }
    emptyEl.hidden = true;
    grid.hidden = false;

    ideas.forEach(function (idea) {
      var color  = DOMAIN_COLORS[idea.domain] || '#6B6660';
      var mailto = 'mailto:hello@houseofideas.in?subject=' +
        encodeURIComponent('Brief request: ' + idea.title);

      var card = document.createElement('article');
      card.className = 'dossier-card';
      card.style.setProperty('--domain-color', color);

      card.innerHTML =
        '<div class="dossier-card__domain">' +
          '<span class="dossier-card__domain-dot"></span>' +
          escapeHtml(idea.domain) +
        '</div>' +
        '<h3 class="dossier-card__title">' + escapeHtml(idea.title) + '</h3>' +
        '<p class="dossier-card__problem">' + escapeHtml(idea.problem) + '</p>' +
        '<div class="dossier-card__signal">' +
          '<span class="dossier-card__signal-label">Why now</span>' +
          '<p class="dossier-card__signal-text">' + escapeHtml(idea.signal) + '</p>' +
        '</div>' +
        '<div class="dossier-card__footer">' +
          '<div class="dossier-card__score">' +
            '<span class="dossier-card__score-label">HoI</span>' +
            '<div class="dossier-card__score-bar">' + buildScoreBar(idea.score, color) + '</div>' +
            '<span class="dossier-card__score-num">' + idea.score + '/10</span>' +
          '</div>' +
          '<a href="' + mailto + '" class="dossier-card__cta">' +
            'Request brief' +
            '<svg viewBox="0 0 14 14" fill="none" aria-hidden="true">' +
              '<path d="M1 7h12M8 2l5 5-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="square"/>' +
            '</svg>' +
          '</a>' +
        '</div>';

      grid.appendChild(card);
    });
  }

  function setFilter(domain) {
    document.querySelectorAll('.catalogue-pill').forEach(function (pill) {
      pill.classList.toggle('is-active', pill.dataset.filter === domain);
    });
    var filtered = domain === 'all'
      ? allIdeas
      : allIdeas.filter(function (i) { return i.domain === domain; });
    renderCards(filtered);
  }

  function buildPills() {
    var domains = Object.keys(DOMAIN_LABELS);

    var allBtn = document.createElement('button');
    allBtn.className = 'catalogue-pill is-active';
    allBtn.dataset.filter = 'all';
    allBtn.textContent = 'All ' + allIdeas.length;
    allBtn.addEventListener('click', function () { setFilter('all'); });
    pillBar.appendChild(allBtn);

    domains.forEach(function (domain) {
      var count = allIdeas.filter(function (i) { return i.domain === domain; }).length;
      if (count === 0) return;

      var btn = document.createElement('button');
      btn.className = 'catalogue-pill';
      btn.dataset.filter = domain;

      var dot = document.createElement('span');
      dot.className = 'catalogue-pill__dot';
      dot.style.background = DOMAIN_COLORS[domain];
      btn.appendChild(dot);
      btn.appendChild(document.createTextNode(DOMAIN_LABELS[domain]));

      btn.addEventListener('click', function () { setFilter(domain); });
      pillBar.appendChild(btn);
    });
  }

  fetch('/data/ideas.json')
    .then(function (r) { return r.json(); })
    .then(function (ideas) {
      allIdeas = ideas;
      buildPills();
      renderCards(ideas);
    })
    .catch(function () {
      emptyEl.hidden = false;
      emptyEl.innerHTML =
        '<p style="color:var(--stone)">Unable to load the catalogue. Please try refreshing.</p>';
    });
})();
