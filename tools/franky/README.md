# Framewise clips on the home server

A clip-making project on `franky` (reached as `ssh home`): drop a video in, get vertical clips out.

```
~/framewise-clips/
  inbox/                     drop recordings here
  inbox/framewise-watch.yaml recipe, style, platform, clip count (default: 3 gaming TikToks of 30 s)
  inbox/output/<video>/      finished clips (reel_1.mp4 ...) + reels.json, or error.txt
  home/                      engine jobs, uploads, API keys
  app/  .venv/               engine source and Python environment
  n8n/                       optional n8n (Docker), talks to the engine on 127.0.0.1:8765
```

## Install or update

From Windows, in the repo:

```bash
powershell -ExecutionPolicy Bypass -File tools/franky/deploy.ps1
```

It copies the engine, installs it in `~/framewise-clips/.venv` and starts the user service
`framewise-clips` (`framewise serve --watch ~/framewise-clips/inbox`, API on 127.0.0.1 only).
Needs Python 3.12+ on the server. Pro (needed for serve + watch folders) starts as the 14-day trial;
for longer, issue a key with `tools/licence_issuer.py` and put it in the service as
`Environment=FRAMEWISE_LICENSE=...`.

## Use

```bash
scp gameplay.mp4 home:framewise-clips/inbox/
ssh home 'ls framewise-clips/inbox/output/gameplay'
ssh home 'journalctl --user -u framewise-clips -f'      # logs
```

One-off runs without the inbox:

```bash
ssh home '~/framewise-clips/.venv/bin/framewise reel ~/framewise-clips/inbox/x.mp4 --style gaming --platform tiktok --set count=5 --out ~/framewise-clips/out'
```

## n8n (optional)

```bash
ssh home 'cd framewise-clips/n8n && AIEDITOR_KEY=<key> docker compose up -d'
ssh -L 5678:127.0.0.1:5678 home        # then open http://127.0.0.1:5678
```

Import `integrations/n8n/long-video-to-shorts.json` or install the `n8n-nodes-framewise` node.
Files under `/clips/inbox` and `/clips/out` in n8n are the server's `inbox/` and `out/` folders.

Only use footage you have the rights to. Game footage (for example GTA) is fine for your own
channels under the publisher's video policy, but not for promoting Framewise on its website.
