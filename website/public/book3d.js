// The book viewer: open it, turn it over, close it. No dependencies.
(function () {
  var dialog = document.getElementById('b3-dialog');
  var opener = document.getElementById('b3-open');
  if (!dialog || !opener || !dialog.showModal) return;
  var view = dialog.querySelector('.b3-view');
  var book = dialog.querySelector('.b3');
  var btn = { open: dialog.querySelector('[data-act="open"]'), turn: dialog.querySelector('[data-act="turn"]'), side: dialog.querySelector('[data-act="side"]') };

  function show(state, side) {
    book.dataset.view = state;
    book.dataset.side = side || 'right';
    view.dataset.view = state;
    btn.open.textContent = state === 'open' ? 'Close the book' : 'Open the book';
    btn.turn.textContent = state === 'back' ? 'Front cover' : 'Back cover';
    btn.side.textContent = book.dataset.side === 'left' ? 'Contents page' : 'Facing page';
    // only the visible faces should be reachable by screen readers
    dialog.querySelector('.b3-in').setAttribute('aria-hidden', state !== 'open');
    dialog.querySelector('.b3-pages').setAttribute('aria-hidden', state !== 'open');
    dialog.querySelector('.b3-back').setAttribute('aria-hidden', state !== 'back');
  }
  opener.addEventListener('click', function () { show('closed'); dialog.showModal(); setTimeout(function () { show('open'); }, 350); });
  btn.open.addEventListener('click', function () { show(book.dataset.view === 'open' ? 'closed' : 'open'); });
  btn.turn.addEventListener('click', function () { show(book.dataset.view === 'back' ? 'closed' : 'back'); });
  btn.side.addEventListener('click', function () { show('open', book.dataset.side === 'left' ? 'right' : 'left'); });
  dialog.querySelector('[data-act="close"]').addEventListener('click', function () { dialog.close(); });
  dialog.addEventListener('click', function (e) { if (e.target === dialog || e.target.classList.contains('b3-stage')) dialog.close(); });
  book.addEventListener('click', function (e) { if (book.dataset.view === 'closed' && !e.target.closest('a')) show('open'); });
})();
