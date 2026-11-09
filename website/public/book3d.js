// The book viewer: open it, turn the pages, turn it over. No dependencies.
(function () {
  var dialog = document.getElementById('b3-dialog');
  var opener = document.getElementById('b3-open');
  if (!dialog || !opener || !dialog.showModal) return;
  var view = dialog.querySelector('.b3-view');
  var book = dialog.querySelector('.b3');
  var leaves = Array.prototype.slice.call(book.querySelectorAll('.b3-leaf')).sort(function (a, b) { return idx(a) - idx(b); });
  var last = book.querySelector('.b3-pages');
  var back = book.querySelector('.b3-back');
  var narrow = window.matchMedia('(max-width: 700px)');
  var act = {};
  dialog.querySelectorAll('[data-act]').forEach(function (b) { act[b.dataset.act] = b; });
  var count = dialog.querySelector('.b3-count');
  var st = { view: 'closed', k: 0, side: 'right' };   // k = which spread is showing once open
  var spreads = leaves.length;

  function idx(el) { return +el.style.getPropertyValue('--i'); }
  function render() {
    var open = st.view === 'open';
    leaves.forEach(function (l, i) { l.classList.toggle('flipped', open && i <= st.k); });
    book.dataset.view = st.view; view.dataset.view = st.view; book.dataset.side = st.side;
    act.open.textContent = open ? 'Close the book' : 'Open the book';
    act.turn.textContent = st.view === 'back' ? 'Front cover' : 'Back cover';
    // only the pages facing the reader are exposed to assistive tech and the tab order
    var leftPage = open ? leaves[st.k].children[1] : null;
    var rightPage = open ? (st.k + 1 < leaves.length ? leaves[st.k + 1].children[0] : last) : null;
    book.querySelectorAll('.b3-paper').forEach(function (pg) {
      var on = pg === leftPage || pg === rightPage || (pg === back && st.view === 'back');
      if (narrow.matches && open) on = on && pg === (st.side === 'left' ? leftPage : rightPage);
      pg.setAttribute('aria-hidden', String(!on)); pg.inert = !on;
    });
    if (narrow.matches) count.textContent = 'Page ' + (st.k * 2 + (st.side === 'right' ? 2 : 1)) + ' of ' + spreads * 2;
    else count.textContent = (st.k + 1) + ' of ' + spreads;
    act.prev.disabled = st.k === 0 && (!narrow.matches || st.side === 'left');
    act.next.disabled = st.k === spreads - 1 && (!narrow.matches || st.side === 'right');
  }
  function go(dir) {
    if (st.view !== 'open') { st.view = 'open'; return render(); }
    if (narrow.matches) {
      if (dir > 0) { if (st.side === 'left') st.side = 'right'; else if (st.k < spreads - 1) { st.k++; st.side = 'left'; } }
      else { if (st.side === 'right') st.side = 'left'; else if (st.k > 0) { st.k--; st.side = 'right'; } }
    } else st.k = Math.max(0, Math.min(spreads - 1, st.k + dir));
    render();
  }
  opener.addEventListener('click', function () {
    st = { view: 'closed', k: 0, side: 'right' }; render(); dialog.showModal();
    setTimeout(function () { st.view = 'open'; render(); }, 350);
  });
  act.open.addEventListener('click', function () { st.view = st.view === 'open' ? 'closed' : 'open'; render(); });
  act.turn.addEventListener('click', function () { st.view = st.view === 'back' ? 'closed' : 'back'; render(); });
  act.prev.addEventListener('click', function () { go(-1); });
  act.next.addEventListener('click', function () { go(1); });
  act.close.addEventListener('click', function () { dialog.close(); });
  dialog.addEventListener('click', function (e) { if (e.target === dialog || e.target.classList.contains('b3-stage')) dialog.close(); });
  dialog.addEventListener('keydown', function (e) { if (e.key === 'ArrowRight') go(1); else if (e.key === 'ArrowLeft') go(-1); });
  book.addEventListener('click', function (e) {
    if (e.target.closest('a')) return;
    if (st.view !== 'open') { st.view = 'open'; return render(); }
    go(e.clientX > window.innerWidth / 2 ? 1 : -1);   // the spread is centred, so the half you click is the page you turn
  });
  narrow.addEventListener('change', render);
})();
