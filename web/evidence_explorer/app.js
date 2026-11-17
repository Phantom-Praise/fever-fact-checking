'use strict';
const $ = id => document.getElementById(id);
const state = {page: 1, total: 0, selected: null, detail: null, view: 'top5', shown: 25};
const modelNames = {calibrated_ensemble: 'Combined model', retrieved_roberta: 'Evidence RoBERTa', claim_only_H2: 'Claim-only H2'};
const patternNames = {wrong_complete:'Wrong label · complete retrieval', wrong_incomplete:'Wrong label · incomplete retrieval', wrong_nei:'NEI predicted as something else', correct_incomplete:'Right label · incomplete retrieval', correct_complete:'Right label · complete retrieval', correct_nei:'Correct NEI'};
let listVersion = 0, detailVersion = 0, articleVersion = 0, timer;
const qualityNames = {complete: 'Complete evidence', partial: 'Some evidence', miss: 'No annotated hits', nei: 'NEI · not scored'};
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function button(text, action, className = '') {
  const node = el('button', text, className); node.type = 'button'; node.addEventListener('click', action); return node;
}
function badge(text, kind = '') { return el('span', text, 'badge ' + kind); }
function number(n) { return n.toLocaleString(); }
function clean(text) {
  return text.replaceAll('-LRB-', '(').replaceAll('-RRB-', ')').replaceAll('-LSB-', '[').replaceAll('-RSB-', ']')
    .replaceAll('-COLON-', ':').replace(/\s+([,.;:!?])/g, '$1').replace(/\(\s+/g, '(').replace(/\s+\)/g, ')');
}
function markedText(text, terms) {
  const p = el('p', null, 'sentence-text'), set = new Set(terms);
  const matches = clean(text).split(/([\p{L}\p{N}_]+)/u);
  for (const part of matches) p.append(set.has(part.toLowerCase()) ? el('mark', part) : document.createTextNode(part));
  return p;
}
function external(info) {
  const a = el('a', 'Wikipedia ↗', 'external'); a.href = info.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a;
}
function notify(text) { $('notification').textContent = text; $('notification').hidden = false; setTimeout(() => $('notification').hidden = true, 4000); }
async function api(url) {
  const r = await fetch(url); const data = await r.json(); if (!r.ok) throw new Error(data.error || 'Could not load data'); return data;
}
async function loadClaims(selectFirst = true) {
  const version = ++listVersion;
  const params = new URLSearchParams({q: $('search').value, quality: $('quality').value,
    correctness: $('correctness').value, label: $('label').value, page: state.page,
    model: $('model').value, prediction: $('prediction').value, outcome: $('outcome').value,
    positive: $('positive').value, diagnostic: $('diagnostic').value});
  $('list-count').textContent = 'Finding claims…';
  try {
    const data = await api('/api/claims?' + params); if (version !== listVersion) return [];
    state.total = data.total; $('claim-list').replaceChildren();
    const active = [];
    if ($('prediction').value !== 'all') active.push('Predicted: ' + $('prediction').value);
    if ($('outcome').value !== 'all') active.push($('outcome').value + ' for ' + $('positive').value);
    if ($('diagnostic').value !== 'all') active.push(patternNames[$('diagnostic').value]);
    $('active-error-filter').hidden = !active.length; $('active-error-filter').textContent = active.join(' · ');
    $('list-count').textContent = number(data.total) + ' matching claims';
    $('page-number').textContent = state.page + ' / ' + Math.max(1, Math.ceil(data.total / data.limit));
    $('prev').disabled = state.page <= 1; $('next').disabled = state.page * data.limit >= data.total;
    if (!data.items.length) $('claim-list').append(el('p', 'No claims match. Try a shorter search or reset the filters.', 'empty'));
    for (const row of data.items) {
      const item = button('', () => loadDetail(row.id), 'claim-item'); item.dataset.id = String(row.id);
      const meta = el('div', null, 'claim-meta'); meta.append(el('span', '#' + row.id), badge(qualityNames[row.quality], row.quality));
      item.append(meta, el('p', row.claim), el('span', row.label + ' · model ' + (row.correct ? 'correct' : 'incorrect'), 'claim-label'));
      $('claim-list').append(item);
    }
    $('claim-list').scrollTop = 0;
    if (selectFirst && data.items.length) await loadDetail(data.items[0].id);
    else document.querySelectorAll('.claim-item').forEach(n => n.classList.toggle('active', n.dataset.id === String(state.selected)));
    if (selectFirst && !data.items.length) {
      ++detailVersion; state.selected = null; state.detail = null; $('detail').replaceChildren(el('p', 'Choose different filters to find a claim.', 'placeholder'));
    }
    return data.items;
  } catch (error) { if (version === listVersion) { $('list-count').textContent = 'Could not load claims'; notify(error.message); } return []; }
}
async function loadDetail(id) {
  const version = ++detailVersion; state.selected = String(id);
  document.querySelectorAll('.claim-item').forEach(n => n.classList.toggle('active', n.dataset.id === String(id)));
  $('detail').replaceChildren(el('p', 'Retrieving and ranking local evidence…', 'placeholder'));
  try {
    const data = await api('/api/claim/' + encodeURIComponent(id) + '?' + new URLSearchParams({model:$('model').value})); if (version !== detailVersion) return;
    state.detail = data; state.view = 'top5'; state.shown = 25;
    history.replaceState(null, '', '#' + id); renderDetail(); $('detail').scrollTop = 0;
  } catch (error) { if (version === detailVersion) $('detail').replaceChildren(el('p', error.message, 'empty')); }
}
function renderDetail() {
  const d = state.detail, root = $('detail'); root.replaceChildren();
  const topline = el('div', null, 'detail-topline');
  topline.append(el('span', 'CLAIM #' + d.id, 'eyebrow'), button('Copy link', async () => {
    try { await navigator.clipboard.writeText(location.href); notify('Claim link copied'); } catch { notify('Copy the claim URL from the address bar.'); }
  }, 'text-button'));
  root.append(topline, el('h2', d.claim));
  const labels = el('div', null, 'label-line');
  labels.append(badge('FEVER: ' + d.label), badge('Model: ' + d.prediction, d.label === d.prediction ? 'complete' : 'miss'));
  root.append(labels);
  root.append(renderTrace(d));
  if (!d.saved_top5_equal) root.append(el('p', 'Current top-five retrieval differs from the saved run. Ranks below are current; model labels and summary statistics belong to the saved evaluation.', 'notice'));
  const coverage = el('div', null, 'coverage ' + d.current_quality);
  const descriptions = {complete: 'The top five contain a complete annotated evidence group.', partial: 'The top five contain some annotated evidence, but not a complete group.', miss: 'The top five contain no annotated evidence for this claim.', nei: 'FEVER marks this claim NEI. There is no required evidence group to score.'};
  coverage.append(el('strong', qualityNames[d.current_quality]), el('p', descriptions[d.current_quality])); root.append(coverage);
  const words = el('div', null, 'query-words'); words.append(el('span', 'Ranking words', 'field-label'));
  for (const term of d.keywords) words.append(el('span', term, 'word-chip')); root.append(words);
  const ignored = el('p', 'Ignored words: ' + (d.ignored_terms.join(', ') || 'none') + '. Scores use the original snapshot text.', 'muted small'); root.append(ignored);
  root.append(renderArticles(d), renderGold(d));
  const heading = el('div', null, 'evidence-heading');
  const title = el('div'); title.append(el('h3', 'Ranked sentences'), el('p', number(d.candidate_count) + ' sentences across ' + d.articles.length + ' candidate articles', 'muted small'));
  heading.append(title); root.append(heading);
  const tabs = el('div', null, 'tabs'); tabs.setAttribute('aria-label', 'Sentence display');
  for (const [value, label] of [['top5', 'Top 5 selected'], ['all', 'All ranked (' + number(d.candidate_count) + ')'], ['annotated', 'Annotated matches']]) {
    const b = button(label, () => { state.view = value; state.shown = 25; renderSentenceList(); }, 'tab');
    b.dataset.view = value; tabs.append(b);
  }
  root.append(tabs, el('p', 'Highlighted words contributed to overlap. Green evidence badges indicate FEVER annotation matches, not a new human judgment.', 'muted small'));
  const container = el('div'); container.id = 'sentences'; root.append(container); renderSentenceList();
}
function renderArticles(d) {
  const details = el('details', null, 'article-details');
  details.append(el('summary', d.articles.length + ' candidate Wikipedia articles · see title-search ranking'));
  details.append(el('p', 'Title query: ' + (d.title_query || '(no search words)') + '. Lower BM25 scores are better here.', 'muted small'));
  const table = el('table', null, 'article-table'), head = el('thead'), tr = el('tr');
  for (const label of ['Rank / article', 'Title BM25', 'Sentences']) tr.append(el('th', label));
  head.append(tr); table.append(head); const body = el('tbody');
  for (const a of d.articles) {
    const row = el('tr'), title = el('td'); title.append(button('#' + a.rank + ' ' + a.title, () => showArticle(a.page_id), 'article-link'));
    row.append(title, el('td', a.bm25_score.toFixed(5), 'mono'), el('td', a.sentence_count)); body.append(row);
  }
  table.append(body); details.append(table); return details;
}
function renderGold(d) {
  const details = el('details', null, 'gold-details');
  details.append(el('summary', 'Compare with FEVER annotations · ' + d.gold_groups.length + ' distinct evidence groups'));
  if (!d.gold_groups.length) details.append(el('p', 'NEI claims have no required evidence group. Retrieved text may still be relevant, but annotation recall is not scored for these claims.', 'muted'));
  else {
    details.append(el('p', 'A full group is required; groups are alternative ways to verify the claim. An unannotated sentence can still be useful.', 'muted small'));
    d.gold_groups.forEach((group, i) => {
      const box = el('div', null, 'gold-group'); box.append(el('h4', 'Evidence group ' + (i + 1)));
      for (const ref of group) {
        const status = ref.selected ? 'In top 5' : ref.candidate ? 'Found below top 5' : ref.text ? 'Outside candidate pages' : 'Not in local index';
        const line = el('div', null, 'reference-line'); line.append(button(ref.title + ' · sentence ' + ref.sentence_id, () => showArticle(ref.page_id), 'article-link'), badge(status, ref.selected ? 'complete' : 'partial'));
        box.append(line, el('p', ref.text ? clean(ref.text) : 'This annotated sentence is unavailable in the local index.'));
      }
      details.append(box);
    });
  }
  return details;
}
function renderSentenceList() {
  const d = state.detail, container = $('sentences'); if (!container) return;
  document.querySelectorAll('.tab').forEach(n => { const active = n.dataset.view === state.view; n.classList.toggle('active', active); n.setAttribute('aria-pressed', String(active)); });
  const rows = state.view === 'top5' ? d.sentences.slice(0, 5) : state.view === 'annotated' ? d.sentences.filter(r => r.annotated) : d.sentences;
  container.replaceChildren();
  if (!rows.length) container.append(el('p', state.view === 'annotated' ? 'No annotated sentences occur in the candidate set. Open the FEVER annotations above to inspect what was missed.' : 'No candidate sentences were retrieved.', 'empty'));
  for (const row of rows.slice(0, state.shown)) {
    const card = el('section', null, 'evidence-card' + (row.selected ? ' selected' : '') + (row.annotated ? ' annotated' : ''));
    const header = el('div', null, 'evidence-card-head'), source = el('div', null, 'source-title');
    source.append(el('span', '#' + row.rank, 'rank'), button(row.title, () => showArticle(row.page_id), 'article-link'), el('span', 'sentence ' + row.sentence_id, 'muted small'));
    header.append(source, el('strong', row.score.toFixed(4), 'score mono')); card.append(header);
    const badges = el('div', null, 'card-badges');
    if (row.selected) badges.append(badge(d.saved_top5_equal ? 'Selected for evidence input' : 'Current top 5', 'selected-badge'));
    if (row.token_exposure) badges.append(badge({full:'All sentence tokens retained',partial:'Partly truncated',none:'Not retained in input'}[row.token_exposure.visibility], row.token_exposure.visibility==='full'?'complete':'partial'));
    if (row.annotated) badges.append(badge('FEVER annotation match', 'complete'));
    if (badges.childNodes.length) card.append(badges);
    card.append(markedText(row.text, row.matched_terms));
    const formula = el('div', null, 'score-breakdown');
    formula.append(el('span', row.overlap + ' shared words'), el('span', '+ ' + row.title_bonus.toFixed(4) + ' article bonus'), el('span', '= ' + row.score.toFixed(4)));
    card.append(formula);
    card.append(el('p', 'Matched: ' + (row.matched_terms.join(', ') || 'none') + ' · article rank #' + row.article_rank, 'muted small'));
    const bottom = el('div', null, 'card-bottom'), raw = el('details', null, 'raw'); raw.append(el('summary', 'Original text'), el('pre', row.text));
    bottom.append(raw, external(row)); card.append(bottom); container.append(card);
  }
  if (rows.length > state.shown) container.append(button('Show next ' + Math.min(25, rows.length - state.shown) + ' sentences (' + (rows.length - state.shown) + ' remaining)', () => { state.shown += 25; renderSentenceList(); }, 'load-more'));
}
function resetFilterValues() {
  clearTimeout(timer); $('search').value='';
  for (const id of ['quality','correctness','label','prediction','outcome','diagnostic']) $(id).value='all';
  state.page=1;
}
function download(name, text, type) {
  const url=URL.createObjectURL(new Blob([text],{type})); const a=el('a'); a.href=url; a.download=name;
  document.body.append(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderAnalysis() {
  const a=state.analysis; if(!a)return;
  const table=el('table',null,'matrix-table'); const head=el('tr'); head.append(el('th','Actual ↓ / predicted →'));
  for(const label of a.labels) head.append(el('th',label==='NOT ENOUGH INFO'?'NEI':label));
  const thead=el('thead');thead.append(head);table.append(thead);const tbody=el('tbody');
  a.labels.forEach((actual,i)=>{
    const tr=el('tr');tr.append(el('th',actual==='NOT ENOUGH INFO'?'NEI':actual));
    a.labels.forEach((predicted,j)=>{
      const td=el('td');const b=button(number(a.matrix[i][j]),()=>{
        resetFilterValues();$('label').value=actual;$('prediction').value=predicted;
        $('prediction-filters').open=true;renderAnalysis();loadClaims();
      },'matrix-cell '+(i===j?'diagonal':'offdiagonal'));
      b.setAttribute('aria-label','Actual '+actual+'; predicted '+predicted+'; '+a.matrix[i][j]+' claims');
      b.classList.toggle('chosen',$('label').value===actual&&$('prediction').value===predicted);
      td.append(b);tr.append(td);
    });tbody.append(tr);
  });
  table.append(tbody);$('matrix').replaceChildren(table);
  $('matrix-metrics').textContent=number(a.n)+' claims · accuracy '+(100*a.accuracy).toFixed(2)+'% · macro-F1 '+a.macro_f1.toFixed(4)+' · strict FEVER '+(100*a.strict_fever_score).toFixed(2)+'%';
  const positive=$('positive').value, p=a.per_class[positive];$('outcome-label').textContent='Outcome for '+positive;
  const explanations={TP:'Correctly predicted '+positive,FP:'Predicted '+positive+', but FEVER says another label',FN:'FEVER says '+positive+', but the model missed it',TN:'Both labels are outside '+positive+' in this binary comparison'};
  $('binary-cards').replaceChildren();
  for(const key of ['TP','FP','FN','TN']){
    const b=button('',()=>{resetFilterValues();$('outcome').value=key;$('prediction-filters').open=true;loadClaims();},'binary-card '+key);
    b.append(el('span',key),el('strong',number(p[key])),el('small',explanations[key]));
    b.setAttribute('aria-label',key+' for '+positive+': '+p[key]+' claims');$('binary-cards').append(b);
  }
  $('binary-metrics').textContent='For '+positive+': precision '+(100*p.precision).toFixed(2)+'% · recall '+(100*p.recall).toFixed(2)+'% · F1 '+p.f1.toFixed(4);
  $('diagnostic-buttons').replaceChildren();
  for(const key of ['wrong_complete','wrong_incomplete','wrong_nei','correct_incomplete']){
    const b=button(patternNames[key]+' · '+number(a.diagnostics[key]),()=>{
      resetFilterValues();$('diagnostic').value=key;$('prediction-filters').open=true;loadClaims();
    },'pattern-button');$('diagnostic-buttons').append(b);
  }
}
async function loadAnalysis() {
  const model=$('model').value; const data=await api('/api/meta?'+new URLSearchParams({model}));
  if(model!==$('model').value)return null;
  state.analysis=data.analysis;
  $('stat-total').textContent=number(data.total);$('stat-recall').textContent=(100*data.complete_recall_at5).toFixed(2)+'%';
  $('stat-denominator').textContent=number(data.verifiable)+' verifiable claims; NEI excluded';
  $('stat-accuracy').textContent=(100*data.accuracy).toFixed(2)+'%';$('stat-model-label').textContent=modelNames[model]+' label accuracy';
  renderAnalysis();return data;
}
function renderTrace(d) {
  const details=el('details',null,'prediction-trace');
  details.open=d.label!==d.prediction||new URL(location.href).searchParams.get('view')==='errors';
  details.append(el('summary','Trace this prediction · '+(d.label===d.prediction?'correct label':'incorrect label')));
  const outcome=el('div',null,'trace-outcomes');
  for(const [label,value]of Object.entries(d.outcomes))outcome.append(badge(value+' for '+(label==='NOT ENOUGH INFO'?'NEI':label),value.startsWith('T')?'complete':'miss'));
  details.append(outcome);
  const narrative={nei:'NEI has no required annotated evidence group. A wrong NEI prediction cannot be assigned to retrieval failure from these annotations alone.',
    article_search:'At least one complete annotated group exists in the local index, but none is in the candidate article set. Article search missed an annotated route to the answer.',
    sentence_ranking:'A complete annotated group is among the candidate sentences, but not in the selected five. The sentence-ranking stage lost it.',
    index_coverage:'No complete annotated group is available in the local index. This is an observed corpus-coverage limitation.',
    input_truncation:'A complete annotated group was selected, but none retained all its sentence tokens after tokenization. The input-length limit removed annotated text; its effect on the label still needs checking.',
    claim_truncation:'The claim itself was shortened by the token limit. Even selected evidence cannot guarantee the model received the complete claim.',
    evidence_reached_input:'A complete annotated group was retrieved and all its tokens survived in the evidence-model input. If the label is wrong, missing annotated evidence is not the observed bottleneck here.',
    selected_unconfirmed_input:'A complete annotated group is in the top five, but tokenizer exposure could not be checked.'};
  if(d.model==='claim_only_H2') details.append(el('p','This is the claim-only model: it never read retrieved evidence. The retrieval and token checks below describe the evidence branch, not the cause of this model’s decision.','notice'));
  if(!d.saved_top5_equal)details.append(el('p','Current retrieval differs from the historical input. The retrieval trace below cannot explain the saved decision reliably.','notice'));
  details.append(el('p',narrative[d.bottleneck]||'Inspect the stages below.','trace-narrative'));
  if(d.model==='calibrated_ensemble'&&d.model_predictions.retrieved_roberta===d.label&&d.prediction!==d.label)
    details.append(el('p','The evidence model predicted the FEVER label correctly, but the score-combination step changed the final label. See the component decisions and combination scores below.','notice'));
  details.append(el('p',d.strict_correct?'This saved prediction passes the official label-and-evidence rule.':'This saved prediction fails the official label-and-evidence rule. A correct label alone is not enough for verifiable claims.','muted small'));
  const steps=el('ol',null,'trace-steps');
  const candidateComplete=d.gold_groups.some(group=>group.length&&group.every(ref=>ref.candidate));
  const selectedComplete=d.current_quality==='complete';
  for(const [title,text]of [
    ['Article and sentence search',d.gold_groups.length?(candidateComplete?'A complete annotated group occurs in the candidate sentences.':'No complete annotated group occurs in the candidate sentences.'):'No annotated reference group for NEI.'],
    ['Top-five selection',d.gold_groups.length?(selectedComplete?'A complete annotated group was selected.':'A complete annotated group was not selected.'):'Evidence coverage is not scored for NEI.'],
    ['Evidence-model token budget',d.token_trace.available?d.token_trace.tokens_before+' tokens before truncation → '+d.token_trace.tokens_after+' non-padding tokens; limit '+d.token_trace.max_length+'.'+(d.token_trace.claim_truncated?' The claim was truncated too.':''):'Tokenizer exposure is unavailable; do not assume all five sentences reached the model.']
  ]){const li=el('li');li.append(el('strong',title),el('p',text));steps.append(li);}
  details.append(steps);
  const caption=el('h4','Saved component decisions');details.append(caption);
  const table=el('table',null,'component-table');const head=el('tr');
  for(const label of ['System','Predicted','P(Supports)','P(Refutes)','P(NEI)'])head.append(el('th',label));
  const thead=el('thead');thead.append(head);table.append(thead);const body=el('tbody');
  for(const model of ['claim_only_H2','retrieved_roberta','calibrated_ensemble']){
    const probabilities=model==='calibrated_ensemble'?null:d.model_probabilities[model==='claim_only_H2'?'claim':'evidence'];
    const row=el('tr');row.className=model===d.model?'selected-component':'';
    if(model===d.model)row.dataset.testid='selected-model';
    row.append(el('td',modelNames[model]),el('td',d.model_predictions[model]));
    for(let i=0;i<3;i++)row.append(el('td',probabilities?(100*probabilities[i]).toFixed(2)+'%':'—'));
    body.append(row);
  }table.append(body);const wrap=el('div',null,'table-scroll');wrap.append(table);details.append(wrap);
  details.append(el('p','The probabilities are model estimates, not verified factual confidence. A high score can still accompany a wrong label.','muted small'));
  const math=el('details',null,'combination-details');math.append(el('summary','Inspect combined decision scores'));
  math.append(el('p','Score = '+d.fusion_config.claim_weight+' × log P(claim model) + '+(1-d.fusion_config.claim_weight).toFixed(2)+' × log P(evidence model), with +'+d.fusion_config.nei_bias+' for NEI. The largest score wins. These are decision scores, not probabilities.','muted small'));
  for(const score of d.fusion_scores)math.append(el('p',score.label+': '+score.claim_contribution.toFixed(4)+' + '+score.evidence_contribution.toFixed(4)+' + '+score.nei_adjustment+' = '+score.score.toFixed(4),'mono small'));
  math.append(el('p',d.fusion_reproduced?'Recalculation matches the saved combined prediction.':'Recalculation differs from the saved prediction; inspect configuration provenance.','muted small'));details.append(math);
  if(d.token_trace.available){
    const tokens=el('details',null,'combination-details');tokens.append(el('summary','Which selected sentences survived tokenization?'));
    for(const sentence of d.token_trace.sentences)tokens.append(el('p','#'+sentence.rank+' · '+sentence.page_id+' / sentence '+sentence.sentence_id+': '+sentence.tokens_kept+' of '+sentence.tokens_total+' tokens retained ('+sentence.visibility+').','small'));
    details.append(tokens);
  }
  details.append(el('p','This trace locates observed gaps. It does not prove a causal explanation: annotations can be incomplete, and no oracle-evidence counterfactual is run here.','muted small'));
  details.append(button('Download this trace (.json)',()=>download('fever-claim-'+d.id+'-trace.json',JSON.stringify(d,null,2),'application/json'),'text-button'));
  return details;
}
async function showArticle(page) {
  const version = ++articleVersion;
  const modal = $('article-modal'); $('modal-title').textContent = 'Loading article…'; $('modal-body').replaceChildren();
  if (!modal.open) modal.showModal();
  try {
    const data = await api('/api/article?' + new URLSearchParams({page_id: page}));
    if (!modal.open || version !== articleVersion) return;
    $('modal-title').textContent = data.title;
    const root = $('modal-body'); root.replaceChildren(); root.append(external(data), el('p', data.sentences.length + ' indexed sentences. IDs are the original zero-based FEVER line numbers.', 'muted small'));
    const d = state.detail;
    for (const sentence of data.sentences) {
      const row = el('section', null, 'article-sentence'); row.append(el('span', sentence.sentence_id, 'mono line-number'), markedText(sentence.text, d ? d.keywords : [])); root.append(row);
    }
    root.scrollTop = 0;
  } catch (error) { if (version === articleVersion) { $('modal-title').textContent = 'Article unavailable'; $('modal-body').replaceChildren(el('p', error.message)); } }
}
$('close-modal').addEventListener('click', () => $('article-modal').close());
$('article-modal').addEventListener('click', e => { if (e.target === $('article-modal')) { const r = e.target.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) e.target.close(); } });
$('search').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { state.page = 1; loadClaims(); }, 250); });
for (const id of ['quality', 'correctness', 'label','prediction','outcome','diagnostic']) $(id).addEventListener('change', () => { state.page = 1; renderAnalysis(); loadClaims(); });
$('prev').addEventListener('click', () => { state.page--; loadClaims(); });
$('next').addEventListener('click', () => { state.page++; loadClaims(); });
$('reset').addEventListener('click', () => { resetFilterValues();renderAnalysis();loadClaims(); });
$('positive').addEventListener('change',()=>{renderAnalysis();if($('outcome').value!=='all'){state.page=1;loadClaims();}});
$('model').addEventListener('change',async()=>{try{resetFilterValues();await loadAnalysis();await loadClaims(false);if(state.selected)await loadDetail(state.selected);}catch(error){notify(error.message);}});
$('analysis-toggle').addEventListener('click',()=>{const open=$('analysis-body').hidden;$('analysis-body').hidden=!open;$('analysis-toggle').setAttribute('aria-expanded',String(open));$('analysis-toggle').textContent=open?'Hide confusion matrix ↑':'Show confusion matrix ↓';});
$('export-confusion').addEventListener('click',()=>{
  const a=state.analysis;if(!a)return;
  const lines=['record_type,model,actual,predicted,count,positive,TP,FP,FN,TN,precision,recall,f1'];
  a.labels.forEach((actual,i)=>a.labels.forEach((predicted,j)=>lines.push(['confusion',a.model,actual,predicted,a.matrix[i][j],'','','','','','','',''].join(','))));
  for(const [label,p]of Object.entries(a.per_class))lines.push(['one_vs_rest',a.model,'','','',label,p.TP,p.FP,p.FN,p.TN,p.precision,p.recall,p.f1].join(','));
  download('fever-'+a.model+'-confusion.csv',lines.join('\n'),'text/csv');
});
(async () => {
  try {
    const data = await loadAnalysis();
    if(new URL(location.href).searchParams.get('view')==='errors')$('analysis-toggle').click();
    if (!data.retriever_source_matches_saved_run) notify('Retriever source has changed since the saved evaluation. Compare current and saved top-five results carefully.');
    const items = await loadClaims(false); const id = location.hash.slice(1) || (items[0] && items[0].id);
    if (id) await loadDetail(id);
  } catch (error) { $('detail').replaceChildren(el('p', 'Could not connect to the local evidence server: ' + error.message, 'empty')); }
})();
