# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Local, dependency-free assets for the module cost explorer."""

STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #21344c; background: #f4f7fb; }
* { box-sizing: border-box; } body { margin: 0; }
main { max-width: 1500px; margin: auto; padding: 2rem; }
h1 { margin: .25rem 0 .6rem; font-size: clamp(1.6rem,3vw,2rem); } h2 { margin-top: 0; }
a { color: #175c97; } a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 3px solid #ad670d; outline-offset: 3px; }
header p { margin: .5rem 0; } .eyebrow { color: #60738b; font-weight: 700; font-size: .8rem; letter-spacing: .07em; }
section, article, .panel { background: white; border: 1px solid #d8e2ed; border-radius: .65rem;
  margin: 1rem 0; padding: 1.1rem; min-width: 0; }
summary { cursor: pointer; padding: .5rem; } summary:hover { background: #edf3f9; }
.tree details details { margin-left: 1rem; border-left: 1px solid #d6e1eb; }
.tree ul { list-style: none; padding-left: 1.5rem; } .call-link { display: inline-block; padding: .3rem; }
.badge { display: inline-block; font-size: .8rem; font-weight: 650; padding: .15rem .4rem;
  border: 1px solid transparent; border-radius: .3rem; white-space: nowrap; }
.complete { color: #155d42; background: #e8f7ee; }
.partial { color: #855414; background: #fff0d7; }
.unavailable { color: #5e435e; background: #f4ecf4; }
.muted, small { color: #60738b; } small { display: block; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .82rem; }
p, h3, li, dd { overflow-wrap: anywhere; } button, input { font: inherit; }
button { cursor: pointer; border: 1px solid #cbd8e5; background: #f3f7fb; color: #25435d;
  padding: .4rem .65rem; border-radius: .3rem; } button:disabled { opacity: .5; cursor: default; }
.table-scroll { overflow-x: auto; } table { width: 100%; border-collapse: collapse; text-align: left; }
caption { text-align: left; margin-bottom: .5rem; color: #60738b; }
th, td { border-bottom: 1px solid #dde4ee; padding: .6rem; vertical-align: top; overflow-wrap: anywhere; }
th { font-size: .85rem; } progress { display: block; width: 100%; height: .6rem; accent-color: #277c6b; }
fieldset { border: 0; padding: .6rem 0; min-width: 0; } legend { font-weight: 650; }
.view-label { display: inline-block; margin: .25rem .7rem .25rem .15rem; }
.metric-panel { display: none; }
#view-module_flops:checked ~ .metric-panel[data-view="module_flops"],
#view-macs:checked ~ .metric-panel[data-view="macs"],
#view-dmas:checked ~ .metric-panel[data-view="dmas"],
#view-parameters:checked ~ .metric-panel[data-view="parameters"],
#view-parameter_bytes:checked ~ .metric-panel[data-view="parameter_bytes"] { display: block; }
.skip { position: absolute; top: -5rem; } .skip:focus { top: .5rem; background: white; padding: .7rem; }
.note { border-left: 3px solid #7096ad; padding-left: .8rem; }
.suggestion { margin: .8rem 0; } .suggestion p { margin: .3rem 0; }
.summary-cards { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 1rem; margin: 1.2rem 0; }
.summary-card { padding: 1rem; background: white; border: 1px solid #d8e2ed; border-radius: .65rem; }
.summary-card .value { margin: .5rem 0; font-size: 1.25rem; font-weight: 700; }
.explorer-grid { display: grid; grid-template-columns: minmax(0,2.5fr) minmax(300px,1fr); align-items: start; gap: 1.2rem; }
.map-column, .inspector { min-width: 0; } .map-column .panel, .inspector { margin-top: 0; }
.map-toolbar { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .5rem; }
.map-scroll { overflow: auto; max-height: 620px; } .module-map { display: block; width: 100%; min-width: 520px; height: auto; }
.module-map a { cursor: pointer; } .module-map a[aria-current="true"] [data-tile] { stroke: #0d584a; stroke-width: 3; }
.module-map a[data-neutral="false"][aria-current="true"] [data-tile] { fill: #237e6b; }
.module-map a[data-neutral="false"][aria-current="true"] text { fill: #fff; }
.module-map a:focus-visible [data-tile] { stroke: #ad670d; stroke-width: 4; }
.module-map [hidden] { display: none; }
.rail-cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap: .6rem; }
.rail-card { display: block; border: 1px solid #d5e0ea; border-radius: .4rem; padding: .7rem;
  text-decoration: none; font-size: .82rem; }
.rail-card.unknown { color: #805019; border-color: #e5bf8c; background: repeating-linear-gradient(135deg,#fff6e8,#fff6e8 7px,#f5e5cb 7px,#f5e5cb 8px); }
.rail-card.zero, .rail-card.small { color: #567087; background: #f3f7fa; }
.rail-card strong { display: block; margin-bottom: .25rem; } .rail-card .badge { margin-top: .3rem; }
.inspector { position: sticky; top: 1rem; } .inspector h3 { margin: .55rem 0; font-size: 1.1rem; }
.inspector .metric-value { font-size: 1.3rem; font-weight: 700; color: #23785f; margin: .8rem 0 .4rem; }
.inspector h4 { margin: 1.2rem 0 .6rem; } .inspector hr { border: 0; border-top: 1px solid #dce4ee; margin: 1rem 0; }
.tensor-shapes { display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; }
.tensor-box { border: 1px solid #dde7ee; border-radius: .4rem; padding: .65rem; background: #f8fbfd; font-size: .8rem; }
.tensor-icon { display: block; position: relative; width: 29px; height: 25px; margin: .4rem .4rem .8rem;
  background: #c8e4eb; border: 1px solid #79aaba; border-radius: 3px; box-shadow: 4px -4px #d7ecf1,8px -8px #e3f1f4; }
.call-card { border: 1px solid #dee7ee; border-radius: .4rem; background: #f6f9fc; padding: .7rem; margin: .6rem 0; font-size: .82rem; }
.call-card p { margin: .3rem 0; } .call-track { height: .45rem; margin: .5rem 0; background: #e5edf3; border-radius: .25rem; }
.call-track span { display: block; height: 100%; background: #429f83; border-radius: .25rem; }
.comparison-note { border-left: 3px solid #70ac95; padding-left: .65rem; font-size: .82rem; }
.appendix { margin: 1.2rem 0; } .appendix > summary { font-weight: 650; }
.under-map { font-size: .8rem; color: #60738b; }
@media (max-width: 950px) { .explorer-grid { grid-template-columns: minmax(0,1fr); } .inspector { position: static; } }
@media (max-width: 650px) { main { padding: 1rem; } th, td { padding: .4rem; }
  .summary-cards { grid-template-columns: minmax(0,1fr); gap: .5rem; } .summary-card { padding: .7rem; } }
@media print { :root { background: white; } main { max-width: none; padding: 0; }
  .metric-panel { display: block !important; } .inspector { position: static; } .map-scroll { max-height: none; }
  section, article { break-inside: avoid; } }
"""

# Report data never enters executable source or innerHTML. Python precomputes
# display strings so large integral counts also keep their exact text in the UI.
SCRIPT = r"""
(() => {
  const data = JSON.parse(document.getElementById('torchscan-data').textContent);
  const views = data.maps;
  let selected = null;
  const collapsed = new Set();
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function activeView() { return document.querySelector('input[name="view"]:checked').id.slice(5); }
  function panel() { return document.querySelector('.metric-panel[data-view="' + activeView() + '"]'); }
  function findNode(id, groupId) {
    const groups = views[activeView()].groups;
    for (const group of groups) {
      if (groupId && group.id !== groupId) continue;
      const node = group.nodes.find(n => n.id === id);
      if (node) return {node, group};
    }
    return null;
  }
  function badge(status) { return el('span', status, 'badge ' + status); }
  function metric(container, text, status, className) {
    const p = el('p', undefined, className);
    const prefix = status + ' · ';
    p.append(badge(status), document.createTextNode(' ' + (text.startsWith(prefix) ? text.slice(prefix.length) : text)));
    container.append(p);
  }
  function inspect(node, group, focus) {
    const aside = panel().querySelector('[data-inspector]');
    aside.replaceChildren();
    aside.append(el('p', 'SELECTED MODULE', 'eyebrow'));
    const heading = el('h3', node.path || '(root)'); heading.tabIndex = -1;
    aside.append(heading, el('p', node.type + ' · ' + node.calls.length + ' observed call(s)', 'muted'));
    metric(aside, node.display, node.status, 'metric-value');
    aside.append(el('p', 'Recorded contribution subtotal; derived from this path and its descendants.', 'under-map'));
    if (node.direct_kind === 'structural') aside.append(el('p', 'No direct estimate was recorded for this container.', 'under-map'));
    if (data.before) {
      const comparison = el('div', undefined, 'comparison-note');
      comparison.append(el('p', 'Before: ' + node.before_display), el('p', 'After: ' + node.display), el('p', node.delta_display));
      aside.append(comparison);
    }
    const sourceCalls = node.calls.length ? node.calls : node.before_calls;
    const sourceReport = node.calls.length ? data.report : data.before;
    if (sourceCalls.length && sourceReport) {
      aside.append(el('h4', node.calls.length ? 'Input / output shapes' : 'Before input / output shapes'));
      const shapes = el('div', undefined, 'tensor-shapes');
      for (const [name, text] of [['Input', sourceCalls[0].input_text], ['Output', sourceCalls[0].output_text]]) {
        const box = el('div', undefined, 'tensor-box');
        const icon = el('span', undefined, 'tensor-icon'); icon.setAttribute('aria-hidden', 'true');
        box.append(icon, el('strong', name), el('p', text)); shapes.append(box);
      }
      aside.append(shapes);
    }
    aside.append(el('h4', 'Call evidence · compute and first attribution'));
    const complete = sourceCalls.filter(c => c.in_group && c.result && c.result.status === 'complete');
    const maximum = Math.max(0, ...complete.map(c => c.result.value));
    for (const call of sourceCalls) {
      const row = el('div', undefined, 'call-card');
      const link = el('a', 'call #' + call.call_index); link.href = '#' + (node.calls.length ? 'call-' : 'before-call-') + call.index;
      row.append(link);
      metric(row, call.display, call.display_status);
      if (call.in_group && call.result && call.result.status === 'complete' && maximum > 0) {
        const track = el('div', undefined, 'call-track'); track.setAttribute('aria-hidden','true');
        const fill = el('span'); fill.style.width = (100 * call.result.value / maximum) + '%'; track.append(fill); row.append(track);
      }
      row.append(el('p', call.parameter_text, 'muted'), el('p', call.input_text + ' → ' + call.output_text, 'muted'));
      aside.append(row);
    }
    if (!node.calls.length) aside.append(el('p', data.before && node.before_calls.length ? 'Removed module; only before calls are recorded.' : 'Structural ancestor; no call record.', 'muted'));
    aside.append(el('hr'), el('p', 'Method: ' + group.method, 'under-map'), el('p', 'Scope: ' + group.scope + ' · unit: ' + group.unit, 'under-map'));
    const diagnostics = data.diagnostics.filter(d => d.path === node.path);
    if (diagnostics.length) {
      aside.append(el('h4', 'Diagnostics for this module path'));
      for (const item of diagnostics) {
        const p = el('p'); const a = el('a', item.code); a.href = '#diagnostic-' + item.index;
        p.append(a, document.createTextNode(': ' + item.message)); aside.append(p);
      }
    }
    const suggestions = data.suggestions.filter(s => node.calls.some(c => s.target === 'call-' + c.index));
    if (suggestions.length) {
      aside.append(el('h4', 'Experiment to try'));
      aside.append(el('p', suggestions[0].experiment, 'under-map'));
    }
    selected = {id: node.id, group: group.id};
    for (const a of panel().querySelectorAll('[data-map-node]')) {
      if (a.dataset.nodeId === node.id && a.dataset.groupId === group.id) a.setAttribute('aria-current','true');
      else a.removeAttribute('aria-current');
    }
    const canBranch = node.children.length > 0;
    panel().querySelector('[data-action="collapse"]').disabled = !canBranch;
    panel().querySelector('[data-action="collapse"]').textContent = collapsed.has(group.id + ':' + node.id) ? 'Expand branch' : 'Collapse branch';
    const tile = Array.from(panel().querySelectorAll('svg a[data-map-node]')).find(a => a.dataset.nodeId === node.id && a.dataset.groupId === group.id);
    panel().querySelector('[data-action="zoom"]').disabled = !canBranch || !tile || tile.hasAttribute('hidden');
    if (focus) heading.focus({preventScroll:true});
  }
  function select(id, groupId, focus = true) {
    const found = findNode(id, groupId); if (found) inspect(found.node, found.group, focus);
  }
  function hiddenByParent(node, group) {
    let parent = group.nodes.find(n => n.id === node.parent);
    while (parent) {
      if (collapsed.has(group.id + ':' + parent.id)) return true;
      parent = group.nodes.find(n => n.id === parent.parent);
    }
    return false;
  }
  function refreshBranches(group) {
    for (const anchor of panel().querySelectorAll('[data-map-node]')) {
      if (anchor.dataset.groupId !== group.id) continue;
      const node = group.nodes.find(n => n.id === anchor.dataset.nodeId);
      anchor.toggleAttribute('hidden', hiddenByParent(node, group));
      anchor.setAttribute('aria-expanded', String(!collapsed.has(group.id + ':' + node.id)));
    }
    for (const gap of panel().querySelectorAll('[data-own-node]')) {
      if (gap.dataset.groupId !== group.id) continue;
      const node = group.nodes.find(n => n.id === gap.dataset.ownNode);
      gap.toggleAttribute('hidden', hiddenByParent(node, group) || collapsed.has(group.id + ':' + node.id));
    }
  }
  function frameBranch(group, node) {
    const svg = panel().querySelector('svg[data-group-id="' + group.id + '"]');
    if (!svg) return;
    if (!node) { svg.setAttribute('viewBox', svg.dataset.originalViewbox); return; }
    const anchor = Array.from(svg.querySelectorAll('[data-map-node]')).find(a => a.dataset.nodeId === node.id);
    if (!anchor || anchor.hasAttribute('hidden')) return;
    const box = anchor.querySelector('[data-tile]').getBBox();
    const original = svg.dataset.originalViewbox.split(' ').map(Number);
    const top = Math.max(0, box.y - 15);
    svg.setAttribute('viewBox', [box.x, top, box.width, original[3] - top].join(' '));
  }
  function reveal() {
    const target = document.getElementById(location.hash.slice(1));
    if (!target) return;
    for (let node = target; node; node = node.parentElement) if (node.tagName === 'DETAILS') node.open = true;
    target.focus({preventScroll:true}); target.scrollIntoView({block:'start'});
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-node-id]');
    if (link && link.tagName.toLowerCase() === 'a') {
      event.preventDefault(); select(link.dataset.nodeId, link.dataset.groupId); return;
    }
    const action = event.target.closest('[data-action]');
    if (action) {
      const found = selected && findNode(selected.id, selected.group); if (!found) return;
      const {node, group} = found;
      const svg = panel().querySelector('svg[data-group-id="' + group.id + '"]');
      if (action.dataset.action === 'collapse') {
        const key = group.id + ':' + node.id;
        if (collapsed.has(key)) collapsed.delete(key); else collapsed.add(key);
        refreshBranches(group); inspect(node, group, false);
      } else if (action.dataset.action === 'zoom') {
        frameBranch(group, node);
      } else if (action.dataset.action === 'up') {
        if (node.parent) {
          const parent = group.nodes.find(n => n.id === node.parent);
          if (svg.getAttribute('viewBox') !== svg.dataset.originalViewbox) frameBranch(group, parent);
          select(node.parent, group.id);
        }
      } else if (action.dataset.action === 'reset') {
        for (const activeGroup of views[activeView()].groups) {
          frameBranch(activeGroup, null);
          for (const key of Array.from(collapsed)) if (key.startsWith(activeGroup.id + ':')) collapsed.delete(key);
          refreshBranches(activeGroup);
        }
        inspect(node, group, false);
      }
      return;
    }
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor && anchor.hash === location.hash) reveal();
  });
  document.addEventListener('keydown', event => {
    const anchor = event.target.closest('a[data-map-node]');
    if (!anchor) return;
    const found = findNode(anchor.dataset.nodeId, anchor.dataset.groupId); if (!found) return;
    const {node, group} = found;
    const anchors = Array.from(panel().querySelectorAll('svg a[data-map-node]')).filter(a => a.dataset.groupId === group.id && !a.hasAttribute('hidden'));
    const ids = new Set(anchors.map(a => a.dataset.nodeId));
    const visible = group.nodes.filter(n => ids.has(n.id) && !hiddenByParent(n,group));
    let next = null;
    if (event.key === 'ArrowRight') {
      collapsed.delete(group.id + ':' + node.id); refreshBranches(group);
      next = node.children.find(id => Array.from(panel().querySelectorAll('svg a[data-map-node]')).some(a => a.dataset.nodeId === id && a.dataset.groupId === group.id && !a.hasAttribute('hidden')));
    } else if (event.key === 'ArrowLeft') next = node.parent;
    else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      const index = visible.findIndex(n => n.id === node.id);
      next = visible[index + (event.key === 'ArrowDown' ? 1 : -1)]?.id;
    } else if (event.key === 'Home') next = visible[0]?.id;
    else if (event.key === 'End') next = visible.at(-1)?.id;
    else if (event.key === ' ') { event.preventDefault(); select(node.id,group.id); return; }
    else return;
    event.preventDefault();
    if (next) {
      const target = Array.from(panel().querySelectorAll('a[data-map-node]')).find(a => a.dataset.nodeId === next && a.dataset.groupId === group.id && !a.hasAttribute('hidden'));
      if (target) { target.focus(); select(next,group.id,false); }
    }
  });
  document.addEventListener('change', event => {
    if (event.target.matches('input[name="view"]')) {
      const found = selected && findNode(selected.id);
      const group = views[activeView()].groups[0];
      const node = found ? found.node : group.nodes.find(n => n.id === panel().dataset.initialNode);
      if (node) inspect(node,found ? found.group : group,false);
    }
  });
  for (const chart of document.querySelectorAll('svg.module-map')) chart.dataset.originalViewbox = chart.getAttribute('viewBox');
  for (const button of document.querySelectorAll('[data-action]')) button.disabled = false;
  const initial = panel().dataset.initialNode;
  if (initial) select(initial, views[activeView()].groups[0].id,false);
  window.addEventListener('hashchange', reveal); reveal();
})();
"""
