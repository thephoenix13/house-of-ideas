/* form.js — submits forms to /api/contact (Vercel serverless function) */

(function () {

  /* Pre-select enquiry type on contact.html from ?type= query param */
  var typeSelect = document.getElementById('cf-type');
  if (typeSelect) {
    var typeMap = {
      fellowship:  'Fellowship Application',
      partnership: 'Partnership Enquiry',
      catalogue:   'Catalogue Access',
      press:       'Press & Media',
      insights:    'General',
    };
    var param = new URLSearchParams(window.location.search).get('type');
    if (param && typeMap[param]) {
      typeSelect.value = typeMap[param];
    }
  }

  function wireForm(opts) {
    var form    = document.getElementById(opts.formId);
    var success = document.getElementById(opts.successId);
    var error   = document.getElementById(opts.errorId);
    var submit  = document.getElementById(opts.submitId);
    if (!form) return;

    var originalLabel = submit.innerHTML;

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      error.classList.remove('is-visible');
      submit.disabled    = true;
      submit.textContent = 'Sending…';

      var data = {};
      var fields = form.querySelectorAll('input, select, textarea');
      fields.forEach(function (el) {
        if (el.name) data[el.name] = el.value;
      });

      fetch('/api/contact', {
        method:  'POST',
        body:    JSON.stringify(data),
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      })
        .then(function (res) {
          if (res.ok) {
            form.style.display = 'none';
            success.removeAttribute('hidden');
            success.classList.add('is-visible');
            success.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          } else {
            throw new Error('server error');
          }
        })
        .catch(function () {
          error.removeAttribute('hidden');
          error.classList.add('is-visible');
          submit.disabled  = false;
          submit.innerHTML = originalLabel;
        });
    });
  }

  wireForm({
    formId:    'contact-form',
    successId: 'form-success',
    errorId:   'form-error',
    submitId:  'cf-submit',
  });

  wireForm({
    formId:    'apply-form',
    successId: 'apply-form-success',
    errorId:   'apply-form-error',
    submitId:  'af-submit',
  });

})();
