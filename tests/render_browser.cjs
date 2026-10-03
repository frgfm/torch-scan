// Optional browser checks: node tests/render_browser.cjs
// Playwright and Chromium are development-only dependencies, never report requirements.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { chromium } = require('playwright');

const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'torchscan-explorer-'));
// Fixed Python source and argument-array invocation avoid shell interpolation.
const generated = spawnSync(process.env.PYTHON || '.venv/bin/python', ['-c', `
import copy
import sys
from pathlib import Path
import torch
from torch import nn
from torchscan import crawl_module, metric_result, render_report

output = Path(sys.argv[1])
model = nn.Sequential(
    nn.Sequential(nn.Linear(4, 8, bias=False), nn.Identity(), nn.Linear(8, 4, bias=False)),
    nn.Linear(4, 2, bias=False),
)
report = crawl_module(model, (4,))
output.joinpath('complete.html').write_text(render_report(report), encoding='utf-8')

mixed = copy.deepcopy(report)
next(layer for layer in mixed['layers'] if layer['path'] == '0.2')['metrics']['module_flops']['method'] = 'zzz_custom_formula'
output.joinpath('mixed.html').write_text(render_report(mixed), encoding='utf-8')

class Sine(nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs)

output.joinpath('baseline.html').write_text(render_report(
    crawl_module(nn.Sequential(nn.Identity()), (4,)), before=crawl_module(nn.Sequential(Sine()), (4,))
), encoding='utf-8')

unknown = copy.deepcopy(report)
for layer in unknown['layers']:
    if layer['path'] == '0' or layer['path'].startswith('0.'):
        layer['metrics']['module_flops'] = metric_result(
            status='unavailable', unit='FLOPs', scope='module_call', method='torchscan_module_formula'
        )
output.joinpath('unknown.html').write_text(render_report(unknown), encoding='utf-8')

payload = '</script><script>window.pwned=1</script><img src="https://example.invalid/pwned" onerror="window.pwned=2">'
hostile = copy.deepcopy(report)
hostile['context']['model_type'] = payload
for layer in hostile['layers']:
    if layer['path'].startswith('0'):
        layer['path'] = payload + layer['path'][1:]
        layer['type'] = payload
hostile['diagnostics'].append(dict(code=payload, severity='warning', metric='flops', path=payload, message=payload))
output.joinpath('hostile.html').write_text(render_report(hostile, title=payload), encoding='utf-8')
output.joinpath('hostile.svg').write_text(render_report(hostile, title=payload, format='svg'), encoding='utf-8')
`, directory], { encoding: 'utf8' });
assert.equal(generated.status, 0, generated.stderr);

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });
  try {
    const context = await browser.newContext({ offline: true });
    const page = await context.newPage();
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(request.url()));
    const panel = () => page.locator('.metric-panel:visible');
    const inspector = () => panel().locator('[data-inspector]');
    const action = name => panel().locator('[data-action="' + name + '"]');
    let data;
    async function load(name) {
      // Managed browsers may block file://; this still runs the complete offline document.
      await page.setContent(fs.readFileSync(path.join(directory, name), 'utf8'));
      data = await page.evaluate(() => JSON.parse(document.getElementById('torchscan-data').textContent));
      assert.equal(await page.locator('.metric-panel:visible').count(), 1);
    }
    function group(view = 'module_flops', method) {
      return data.maps[view].groups.find(item => !method || item.method === method);
    }
    function chartNode(modulePath, currentGroup = group()) {
      const node = currentGroup.nodes.find(item => item.path === modulePath);
      assert.ok(node, 'fixture path exists: ' + modulePath);
      return panel().locator('svg a[data-map-node][data-node-id="' + node.id + '"][data-group-id="' + currentGroup.id + '"]');
    }
    function railNode(modulePath, currentGroup = group()) {
      const node = currentGroup.nodes.find(item => item.path === modulePath);
      return panel().locator('.rail-card[data-node-id="' + node.id + '"][data-group-id="' + currentGroup.id + '"]');
    }
    async function selectedPath() { return inspector().locator('h3').textContent(); }
    async function assertTileFits(locator, focused = true) {
      if (focused) assert.equal(await locator.evaluate(node => node === document.activeElement), true);
      assert.equal(await locator.locator('[data-tile]').evaluate(tile => {
        const box = tile.getBBox();
        const view = tile.ownerSVGElement.viewBox.baseVal;
        return box.x >= view.x && box.y >= view.y
          && box.x + box.width <= view.x + view.width + .001
          && box.y + box.height <= view.y + view.height + .001;
      }), true, 'module fits inside its SVG viewport');
    }

    await load('complete.html');
    await chartNode('0.0').focus();
    await page.keyboard.press('ArrowDown');
    assert.equal(await selectedPath(), '0.2', 'arrow navigation skips the zero-width Identity');
    assert.equal(await chartNode('0.2').evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('Space');
    assert.equal(await inspector().locator('h3').evaluate(node => node === document.activeElement), true);

    // Selection fill follows the inspector instead of sticking to the initially costly module.
    await chartNode('0.2').click();
    const selectedFill = await chartNode('0.2').locator('[data-tile]').evaluate(node => getComputedStyle(node).fill);
    await chartNode('1').click();
    assert.equal(await chartNode('1').getAttribute('aria-current'), 'true');
    assert.equal(await chartNode('1').locator('[data-tile]').evaluate(node => getComputedStyle(node).fill), selectedFill);
    assert.notEqual(await chartNode('0.2').locator('[data-tile]').evaluate(node => getComputedStyle(node).fill), selectedFill);

    await chartNode('0').click();
    await action('collapse').click();
    assert.equal(await chartNode('0.0').isVisible(), false);
    assert.equal(await railNode('0.1').isVisible(), true, 'status rail stays available when a branch is collapsed');
    assert.equal(await action('collapse').textContent(), 'Expand branch');
    await action('reset').click();
    assert.equal(await chartNode('0.0').isVisible(), true);
    assert.equal(await action('collapse').textContent(), 'Collapse branch');

    const svg = panel().locator('svg.module-map').first();
    const originalViewBox = await svg.getAttribute('viewBox');
    await action('zoom').click();
    assert.notEqual(await svg.getAttribute('viewBox'), originalViewBox);
    await action('up').click();
    assert.equal(await selectedPath(), '(root)');
    await assertTileFits(chartNode(''), false);

    // Zoom must not leave keyboard targets outside the displayed SVG viewport.
    for (const key of ['End', 'Home', 'ArrowLeft']) {
      await chartNode('0').click();
      await action('zoom').click();
      await chartNode('0').focus();
      await page.keyboard.press(key);
      const expected = key === 'End' ? '1' : '';
      assert.equal(await selectedPath(), expected || '(root)');
      await assertTileFits(chartNode(expected));
    }
    await chartNode('0').click();
    await action('zoom').click();
    await chartNode('0.2').focus();
    await page.keyboard.press('Tab');
    await assertTileFits(chartNode('1'));
    assert.equal(await selectedPath(), '0', 'Tab reveals focus without selecting the module');
    await page.keyboard.press('Enter');
    assert.equal(await selectedPath(), '1');

    // Metric switching preserves module selection, and native radios work from the keyboard.
    await chartNode('0.0').click();
    await page.locator('#view-parameters').check();
    assert.equal(await panel().getAttribute('data-view'), 'parameters');
    assert.equal(await selectedPath(), '0.0');
    assert.equal(await chartNode('0.0', group('parameters')).getAttribute('aria-current'), 'true');
    await page.locator('#view-module_flops').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await panel().getAttribute('data-view'), 'macs');

    // Evidence links reveal their nested details and transfer keyboard focus.
    await page.locator('#view-module_flops').check();
    await chartNode('0.0').click();
    const evidence = inspector().locator('a[href^="#call-"]').first();
    const evidenceId = (await evidence.getAttribute('href')).slice(1);
    await evidence.focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(id => document.getElementById(id).open, evidenceId);
    assert.equal(await page.evaluate(() => document.activeElement.id), evidenceId);
    assert.equal(await page.locator('#' + evidenceId + ' pre').first().isVisible(), true);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.setViewportSize({ width: 1280, height: 900 });

    // One reset restores every incompatible method group's independent branch state.
    await load('mixed.html');
    const custom = group('module_flops', 'zzz_custom_formula');
    await chartNode('0.2', custom).click();
    await page.locator('#view-parameters').check();
    await page.locator('#view-module_flops').check();
    assert.equal(await chartNode('0.2', custom).getAttribute('aria-current'), 'true');
    assert.match(await inspector().locator('.metric-value').textContent(), /complete.*60 FLOPs/);
    assert.match(await inspector().textContent(), /Method: zzz_custom_formula/);
    await load('mixed.html');
    await page.locator('#view-parameters').check();
    await chartNode('0.2', group('parameters')).click();
    await page.locator('#view-module_flops').check();
    assert.equal(await chartNode('0.2', group('module_flops', 'zzz_custom_formula')).getAttribute('aria-current'), 'true',
      'a newly selected module prefers its recorded method group over a placeholder');
    for (const currentGroup of data.maps.module_flops.groups) {
      await chartNode('', currentGroup).click();
      await action('collapse').click();
      assert.equal(await chartNode('0', currentGroup).isVisible(), false);
    }
    await action('reset').click();
    for (const currentGroup of data.maps.module_flops.groups) {
      assert.equal(await chartNode('0', currentGroup).isVisible(), true);
    }
    assert.equal(await action('collapse').textContent(), 'Collapse branch');

    await load('baseline.html');
    await chartNode('0').click();
    assert.match(await inspector().textContent(), /Before diagnostics/);
    const beforeWarnings = inspector().locator('a[href^="#before-diagnostic-"]');
    assert.ok(await beforeWarnings.count() >= 2);
    assert.match((await beforeWarnings.allTextContents()).join(' '), /Global.*uncounted_operator/);
    await beforeWarnings.last().click();
    assert.equal(await page.locator('#' + (await beforeWarnings.last().getAttribute('href')).slice(1)).isVisible(), true);

    // An unavailable ancestor has descendants but no numerical rectangle to zoom into.
    await load('unknown.html');
    assert.equal(await chartNode('0').count(), 0);
    await railNode('0').click();
    assert.match(await inspector().textContent(), /unavailable.*unknown/s);
    assert.equal(await action('zoom').isDisabled(), true);

    // Hostile names survive both initial serialization and later inspector DOM construction.
    await load('hostile.html');
    const hostilePath = data.maps.module_flops.groups[0].nodes.find(node => node.path && node.path !== '1').path;
    await chartNode(hostilePath).click();
    assert.equal(await selectedPath(), hostilePath);
    await page.locator('#view-parameters').check();
    assert.equal(await selectedPath(), hostilePath);
    assert.equal(await page.evaluate(() => window.pwned), undefined);
    assert.equal(await page.locator('img').count(), 0);
    await page.setContent(fs.readFileSync(path.join(directory, 'hostile.svg'), 'utf8'));
    assert.equal(await page.evaluate(() => window.pwned), undefined);
    assert.equal(await page.locator('svg script, svg image, svg foreignObject').count(), 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    await context.close();

    const noScript = await browser.newContext({ javaScriptEnabled: false, offline: true });
    const fallback = await noScript.newPage();
    await fallback.setContent(fs.readFileSync(path.join(directory, 'complete.html'), 'utf8'));
    await fallback.locator('#view-parameters').check();
    assert.equal(await fallback.locator('.metric-panel:visible').getAttribute('data-view'), 'parameters');
    assert.equal(await fallback.locator('.metric-panel:visible svg.module-map').isVisible(), true);
    assert.equal(await fallback.locator('.metric-panel:visible [data-action="zoom"]').isDisabled(), true);
    const appendix = fallback.locator('details.appendix').filter({ has: fallback.locator('section#totals') });
    await appendix.locator(':scope > summary').click();
    const hierarchy = fallback.locator('.tree > details');
    await hierarchy.locator(':scope > summary').focus();
    await fallback.keyboard.press('Space');
    assert.equal(await hierarchy.getAttribute('open'), null);
    await fallback.keyboard.press('Enter');
    assert.notEqual(await hierarchy.getAttribute('open'), null);
    await fallback.locator('#call-2 > summary').click();
    assert.equal(await fallback.locator('#call-2 pre').first().isVisible(), true);
    await noScript.close();
    console.log('Explorer browser checks passed: keyboard, unknown rails, selection, collapse/reset, parent zoom, grouped methods, views, evidence, offline/mobile, hostile DOM/SVG, JavaScript-disabled fallback.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => fs.rmSync(directory, { recursive: true, force: true }));
