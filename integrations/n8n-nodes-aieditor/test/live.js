// Runs the compiled AIEditor n8n node against a live engine with a fake IExecuteFunctions.
// usage: npm run build && node test/live.js <baseUrl> <apiKey> <stockDir>   (needs `aieditor serve` running)
const fs = require('fs');
const path = require('path');
const [, , BASE, KEY, STOCK] = process.argv;
const nodeJs = path.join(__dirname, '../dist/nodes/AIEditor/AIEditor.node.js');
const { AIEditor } = require(path.resolve(nodeJs));

async function httpRequest(opts) {
	let url = opts.url;
	if (opts.qs) url += '?' + new URLSearchParams(Object.entries(opts.qs).map(([k, v]) => [k, String(v)]));
	const headers = { ...(opts.headers || {}) };
	let body = opts.body;
	if (body !== undefined && !Buffer.isBuffer(body) && typeof body === 'object') {
		body = JSON.stringify(body);
		headers['Content-Type'] = 'application/json';
	}
	const res = await fetch(url, { method: opts.method || 'GET', headers, body });
	if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
	if (opts.encoding === 'arraybuffer') return Buffer.from(await res.arrayBuffer());
	const text = await res.text();
	return opts.json ? JSON.parse(text) : text;
}

function ctx(items, params, { continueOnFail = false } = {}) {
	const node = new AIEditor();
	const props = Object.fromEntries(node.description.properties.map((p) => [p.name, p.default]));
	return {
		getInputData: () => items,
		getCredentials: async () => ({ baseUrl: BASE + '/', apiKey: KEY }),
		getNodeParameter: (name, i, fallback) => (name in params ? params[name] : name in props ? props[name] : fallback),
		getNode: () => ({ name: 'AIEditor', type: 'aiEditor', typeVersion: 1, parameters: params }),
		continueOnFail: () => continueOnFail,
		helpers: {
			httpRequestWithAuthentication: async function (cred, opts) {
				if (cred !== 'aiEditorApi') throw new Error('bad credential name ' + cred);
				return httpRequest({ ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${KEY}` } });
			},
			assertBinaryData: (i, prop) => {
				const b = items[i].binary?.[prop];
				if (!b) throw new Error(`no binary ${prop}`);
				return b;
			},
			getBinaryDataBuffer: async (i, prop) => Buffer.from(items[i].binary[prop].data, 'base64'),
			prepareBinaryData: async (buf, fileName) => ({ data: buf.toString('base64'), fileName, size: buf.length }),
		},
		run: () => node.execute.call(null),
		node,
	};
}
async function exec(items, params, o) {
	const c = ctx(items, params, o);
	return (await c.node.execute.call(c))[0];
}
const bin = (file) => ({ json: {}, binary: { data: { data: fs.readFileSync(file).toString('base64'), fileName: path.basename(file) } } });
const ok = (cond, msg) => { if (!cond) { console.error('FAIL:', msg); process.exitCode = 1; } else console.log('ok:', msg); };

(async () => {
	const t0 = Date.now();
	// 1. Edit Photo: binary in, binary out
	const photos = await exec([bin(path.join(STOCK, 'food.jpg')), bin(path.join(STOCK, 'portrait.jpg'))],
		{ operation: 'editPhoto', style: 'food', platform: 'instagram_post', prompt: 'warm and bright' });
	ok(photos.length === 2, 'editPhoto returns one item per input item');
	for (const [k, it] of photos.entries()) {
		ok(it.json.state === 'done', `photo ${k} job done (${it.json.recipe})`);
		ok(it.pairedItem.item === k, `photo ${k} pairedItem`);
		const f = it.binary?.file_1;
		ok(f && f.size > 10000 && /\.jpg$/.test(f.fileName), `photo ${k} binary file_1 ${f?.fileName} ${f?.size}B`);
		if (f && process.env.OUT_DIR) fs.writeFileSync(path.join(process.env.OUT_DIR, `n8n_${k}_${f.fileName}`), Buffer.from(f.data, 'base64'));
	}
	console.log(`  photos took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

	// 2. Get Job without download
	const got = await exec([{ json: {} }], { operation: 'getJob', jobId: photos[0].json.id, download: false });
	ok(got[0].json.id === photos[0].json.id && got[0].json.state === 'done' && !got[0].binary, 'getJob returns job json only');

	// 3. Transcribe via file id refs mode (upload done outside the node)
	const up = await httpRequest({ method: 'PUT', url: BASE + '/v1/files', qs: { name: 'interview.webm' },
		headers: { Authorization: `Bearer ${KEY}` }, body: fs.readFileSync(path.join(STOCK, 'interview.webm')), json: true });
	const t1 = Date.now();
	const tr = await exec([{ json: {} }], { operation: 'transcribe', inputMode: 'refs', refs: up.file_id });
	const names = Object.values(tr[0].binary || {}).map((b) => b.fileName).sort();
	ok(tr[0].json.state === 'done' && names.some((n) => n.endsWith('.srt')) && names.some((n) => n.endsWith('.txt')),
		`transcribe outputs ${names.join(', ')} in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
	const txt = Object.values(tr[0].binary).find((b) => b.fileName.endsWith('.txt'));
	if (txt) console.log('  transcript start:', Buffer.from(txt.data, 'base64').toString('utf8').slice(0, 120).replace(/\s+/g, ' '));

	// 4. Make Reels without waiting, then Cancel Job
	const queued = await exec([{ json: {} }], { operation: 'makeReels', inputMode: 'refs', refs: up.file_id, wait: false, platform: 'tiktok', extra: { count: 1, length: 15 } });
	ok(['queued', 'running'].includes(queued[0].json.state) && !queued[0].binary, `makeReels no-wait returns ${queued[0].json.state}`);
	const cancelled = await exec([{ json: {} }], { operation: 'cancelJob', jobId: queued[0].json.id });
	const c0 = cancelled[0].json;
	ok(c0.state === 'cancelled' || c0.stage === 'Cancelling', `cancelJob -> ${c0.state} / ${c0.stage}`);
	const after = await httpRequest({ url: `${BASE}/v1/jobs/${c0.id}/wait`, qs: { timeout: 60 }, headers: { Authorization: `Bearer ${KEY}` }, json: true });
	ok(after.state === 'cancelled', `cancelled job ends as ${after.state}`);

	// 5. Errors: continueOnFail gives an error item; otherwise throws
	const bad = await exec([{ json: {} }], { operation: 'renderProgram', inputMode: 'refs', refs: up.file_id, program: 'this is not a program' }, { continueOnFail: true });
	ok(typeof bad[0].json.error === 'string', `continueOnFail error item: ${String(bad[0].json.error).slice(0, 90)}`);
	let threw = false;
	try { await exec([{ json: {} }], { operation: 'editPhoto', inputMode: 'refs', refs: '' }); } catch (e) { threw = /No input files/.test(e.message); }
	ok(threw, 'empty input throws NodeOperationError');
	let badKey = false;
	try {
		await httpRequest({ url: BASE + '/v1/me', headers: { Authorization: 'Bearer aie_00000000_nope' }, json: true });
	} catch (e) { badKey = /401/.test(e.message); }
	ok(badKey, 'credential test endpoint rejects a bad key');
	const me = await httpRequest({ url: BASE + '/v1/me', headers: { Authorization: `Bearer ${KEY}` }, json: true });
	ok(me && me.scopes, `credential test endpoint /v1/me ok: ${JSON.stringify(me.scopes)}`);
	console.log(process.exitCode ? 'SOME FAILED' : 'ALL PASSED', `total ${((Date.now() - t0) / 1000).toFixed(1)}s`);
})().catch((e) => { console.error('CRASH', e); process.exit(1); });
