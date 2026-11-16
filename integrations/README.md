# Automating AIEditor

Everything runs on your own computer. Start the engine without the desktop app:

```bash
aieditor serve                 # http://127.0.0.1:8765, interactive docs at /docs
```

The first start creates an API key and prints it once. Make more with
`aieditor keys create NAME [--scope jobs] [--scope local] [--scope admin]`.

| Scope | Allows |
|---|---|
| `jobs` | upload inputs, create / read / cancel jobs, download results |
| `local` | use file paths on this computer as inputs, and `output_dir` |
| `admin` | manage keys, read the webhook secret |

## Recipes

| Recipe | In | Out |
|---|---|---|
| `reel` | one long video (+ optional music) | `reel_N.mp4` vertical clips, best moments first, pauses cut, speaker kept in frame, captions; `reels.json` |
| `highlights` | a video | `highlights.json` (start, end, title, reason, score), optionally `highlight_N.mp4` |
| `captions` | a video | `<name>_captioned.mp4` and `<name>.srt` |
| `video_edit` | videos, photos, music + `prompt` | `edit_N.mp4`, `edits.json` |
| `photo_edit` | photos (+ optional `prompt`) | `<name>_edit.jpg` per photo and version |
| `render_program` | anything + `options.program` (Edit DSL) | `render.mp4` or one photo per input |
| `transcribe` | video / audio | `.srt`, `.txt`, word-timed `.json` |

Every job also takes the prompt options: `style` (cinematic, energetic_reel, vlog, travel, food, ...) and
`output` (`platform`, `length`, `versions`, `captions`, `music`, `quality: draft|standard|high`).
`GET /v1/recipes` and `GET /v1/options` list everything with its schema.

## HTTP

```bash
curl -X PUT "http://127.0.0.1:8765/v1/files?name=talk.mp4" -H "Authorization: Bearer $KEY" --data-binary @talk.mp4
# -> {"file_id": "f_..."}

curl -X POST http://127.0.0.1:8765/v1/jobs -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"recipe":"reel","inputs":["f_..."],"style":"energetic_reel","output":{"platform":"tiktok","length":30},"options":{"count":3},"webhook":"https://example.com/hook"}'
# -> 202 {"id": "j_...", "state": "queued", ...}

curl "http://127.0.0.1:8765/v1/jobs/j_.../wait?timeout=120" -H "Authorization: Bearer $KEY"   # long-poll
curl -O "http://127.0.0.1:8765/v1/jobs/j_.../files/reel_1.mp4" -H "Authorization: Bearer $KEY"
```

Inputs can be uploaded file ids, `http(s)` URLs (downloaded by the engine), or paths (`local` scope).
`DELETE /v1/jobs/{id}` cancels a running job or deletes a finished one. Finished jobs and uploads are
removed after 7 days (`aieditor serve --keep-days N`).

## Webhooks

When a job finishes, the engine POSTs `{"event": "job.done" | "job.failed" | "job.cancelled", "job": {...}}` with
`X-AIEditor-Signature: t=<unix>,v1=<hex>` where `v1 = HMAC_SHA256(secret, "<t>." + raw body)`.
The secret is the job's `webhook.secret`, or the engine's (`aieditor keys secret`). Failed deliveries are
retried after 2, 10 and 30 s. Verify it over the **raw** body. In n8n: turn on the Webhook node's
*Raw Body* option, allow `crypto` (`NODE_FUNCTION_ALLOW_BUILTIN=crypto`), then in a Code node:

```js
const crypto = require('crypto');
const raw = await this.helpers.getBinaryDataBuffer(0, 'data');
const [t, v1] = $input.first().json.headers['x-aieditor-signature'].split(',').map(p => p.split('=')[1]);
const mac = crypto.createHmac('sha256', $env.AIEDITOR_WEBHOOK_SECRET).update(`${t}.`).update(raw).digest('hex');
if (mac !== v1 || Math.abs(Date.now() / 1000 - Number(t)) > 300) throw new Error('bad signature');
return [{ json: JSON.parse(raw.toString('utf8')) }];
```

## n8n

- **No install:** import [`n8n/long-video-to-shorts.json`](n8n/long-video-to-shorts.json). Set `AIEDITOR_KEY`
  (and optionally `AIEDITOR_URL`, `AIEDITOR_INPUT`, `AIEDITOR_OUTPUT`) in n8n's environment. It uploads a video,
  makes 3 TikTok reels, waits and saves the files. n8n in Docker on the same PC reaches the engine at
  `http://host.docker.internal:8765`.
  n8n 2.x blocks `$env` in expressions and limits file access by default, so for this template set
  `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` and add your input and output folders to `N8N_RESTRICT_FILE_ACCESS_TO`
  (separated by `;`). It asks for 3 reels of 30 s; a short video gives fewer, since each reel needs its own
  strong moment.
- **Community node:** [`n8n-nodes-aieditor`](n8n-nodes-aieditor) adds an *AIEditor* node (Make Reels, Find
  Highlights, Add Captions, Edit Video, Edit Photo, Render Program, Transcribe, Get Job, Cancel Job) that uploads
  binary data from the previous node, waits, and returns the results as binary `file_1..n`.
  Build: `npm install && npm run build`, then install the folder in n8n (Settings > Community nodes, or
  `npm install <path>` inside `~/.n8n/custom`, or point `N8N_CUSTOM_EXTENSIONS` at the built `dist` folder). `npm run test:live -- http://127.0.0.1:8765 <key> <media dir>`
  runs every operation of the built node against a running engine without n8n (the media dir needs
  `food.jpg`, `portrait.jpg` and `interview.webm`). Tested with n8n 2.47.7: the node, its credential test and
  the HTTP template all run there.

## Command line

```bash
aieditor reel talk.mp4 --platform tiktok --set count=3 --out shorts/
aieditor edit a.mp4 b.mp4 song.mp3 --prompt "fun travel recap" --style travel --versions 2
aieditor photo *.jpg --style food --platform instagram_post
aieditor captions talk.mp4 --set style=boxed
aieditor render a.mp4 --program edit.vdsl
aieditor transcribe interview.mp4
```

One-shot commands run without a server and print a JSON summary (with the result paths) on stdout.

## Watch folders

```bash
aieditor watch ~/Videos/inbox --recipe reel --platform tiktok
aieditor serve --watch ~/Videos/inbox            # or together with the API
```

Each new file (once it stops growing) is processed; results go to `inbox/output/<file name>/`, errors to
`error.txt` there. Put an `aieditor-watch.yaml` in the folder to choose the recipe and options:

```yaml
recipe: reel
style: energetic_reel
output: {platform: tiktok, length: 30}
options: {count: 3}
```

## AI agents (MCP)

```json
{"mcpServers": {"aieditor": {"command": "aieditor", "args": ["mcp"]}}}
```

Tools: `list_recipes`, `list_styles`, `run_recipe` (local paths in, local paths out), `get_job`, `cancel_job`.

## Network use

`aieditor serve` only listens on 127.0.0.1. `--host 0.0.0.0 --allow-lan` opens it to your network (with a
warning). Give other machines keys without the `local` scope, and never expose the port to the internet
directly; use a tunnel that adds its own authentication.
