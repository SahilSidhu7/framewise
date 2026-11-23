# Framewise: how it is sustained and how it gets users

Positioning: **"The free, open-source CapCut / Photoshop / After Effects alternative. Runs on your PC, with local AI."**

Decided 2026-11-23: everything is free and open source under AGPL-3.0. There is no Pro tier, no licence key and no
feature held back. The project is funded by people who want it to exist and by optional extras that cost nothing to
deliver, never by locking the editor.

## 1. Market in short
- Video editing software is roughly a $3.5–4.7B market in 2026 (third-party estimates disagree), growing about 6% a year.
  CapCut has 300M+ monthly users; Adobe owns professional photo and motion work by subscription.
- Open gaps:
  - **Free, local, private, with good AI.** Open-source editors (OpenCut, Kdenlive, Shotcut, GIMP, Krita, Natron) have
    weak or no AI; CapCut raised its Pro price and its terms are widely disliked; Adobe's subscription is the most
    common complaint in creative communities.
  - **One tool for photo, video and compositing.** People bounce between CapCut, Photoshop and After Effects. Nothing
    open source covers all three with a shared layer model.
  - **Automation.** No mainstream editor has a local API. Cloud clipping tools (OpusClip and similar) charge per minute.
- Order of work: clipping (highlights, reels, captions) and AI photo edits now; then layers and GPU compositing, which
  unlock Photoshop-class photo work and After Effects-class motion work; then VFX.

## 2. What is free
Everything in this repository: the photo editor, the video editor, clipping, captions, AI photo edits, layers and
compositing as they land, the CLI, the local API, MCP and the n8n node. No watermark, no account, no usage limits.
Source under AGPL-3.0, so nobody can take it closed.

## 3. How the project is sustained
| Source | What | Our cost |
|---|---|---|
| **GitHub Sponsors** | Monthly or one-time sponsorship of the maintainer. Sponsor tiers get a name in the README and release notes, and a vote on the roadmap | 0 |
| **Open Collective** | Transparent fund for shared costs (code signing, domain, CI minutes, test hardware). Companies that use Framewise can sponsor here and get an invoice | 0 (fees per donation) |
| **Packs (optional, paid)** | Style/look packs, VFX packs (overlays, particles, transitions), music and SFX packs. Content, not code: the editor never needs them, and anyone can make and share free packs in the same format. Later a marketplace with revenue share for creators | 0 (file download) |
| **Hosted render (later, optional, pay per use)** | For people with weak PCs: the same open-source engine runs a job on a rented GPU. Prepaid credits, priced at 3–4x GPU cost | pay-per-second, scale-to-zero serverless GPU; no job, no bill |

Rule: **no server of ours runs AI unless someone has already paid for that exact job.** The free product costs us
nothing to run, because it runs on the user's PC.

Prices for packs and credits are starting points to test, not commitments.

## 4. Cloud without paying for cloud
In the order to use them:
1. **Local first.** The free product costs us nothing per user.
2. **The user's PC is the cloud.** Reach your own engine from a phone or cloud n8n through a tunnel you run yourself.
3. **Bring your own.** The user pastes their own API key, or launches a one-click template on their own GPU-rental
   account and pays that provider directly.
4. **Prepaid serverless** for hosted render: credits are bought before use, so revenue arrives before cost. Never an
   always-on GPU server.
5. **Free credits** for development: open-source and startup programmes from NVIDIA, Microsoft, Google and AWS, plus
   serverless providers' free monthly credit. Check current terms when applying.

Unavoidable costs: a domain (about $12/year); Windows code signing (free programmes exist for open-source projects,
e.g. SignPath Foundation; otherwise about $10/month); payment fees on packs and credits. Installers and model downloads
go through GitHub Releases and Hugging Face at no cost.

## 5. Payments
- A merchant of record (Lemon Squeezy, Paddle or Gumroad) sells packs and render credits, handles worldwide sales tax and
  pays out. No monthly fee, roughly 5% plus fees per sale. Works from India without a foreign company.
- Donations go through GitHub Sponsors (no fee for personal accounts) and Open Collective (via a fiscal host).
- Packs are plain files. There is no licence check in the app.

## 6. Marketing (budget: time, not money)
Headline everywhere: **"Free, open-source CapCut / Photoshop / After Effects alternative."**
1. **Open source is the launch.** GitHub repo with a strong README, GIFs and a clear roadmap; Show HN; Product Hunt;
   AlternativeTo entries against CapCut, Photoshop, After Effects and OpusClip; r/opensource, r/selfhosted,
   r/LocalLLaMA, r/VideoEditing and r/photography.
2. **Comparison pages** for "free CapCut alternative", "open-source Photoshop alternative", "After Effects alternative"
   and "OpusClip alternative" searches. Honest about what is done and what is coming.
3. **Dogfood content.** Every demo, reel and short is made with the tool. Before/after clips are the ad.
4. **Automation crowd.** n8n workflow templates ("YouTube video in, 5 captioned shorts out"), the community node,
   r/n8n and the n8n forum.
5. **Contributors as marketing.** Good-first-issue labels, a public roadmap, credit in release notes. Pack creators
   share their packs, which shows the editor.
6. **VFX wave** once compositing lands: "my 10-year-old made this" superpower clips on Shorts, TikTok and Reels; a
   classroom guide for teachers.
7. **Measure** with opt-in, privacy-respecting counts only: downloads, GitHub stars, sponsors, pack sales. First 3
   months after release: aim for 2,000 downloads, 1,000 stars and 25 sponsors.

## 7. Risks
| Risk | Answer |
|---|---|
| Donations alone rarely pay a full-time maintainer | packs and hosted render add income that does not lock features; keep running costs near zero |
| Needs a decent PC; casual users edit on phones | phone as a remote for the PC, hosted render credits, a Lite hardware tier |
| Someone forks it and sells it | AGPL-3.0 keeps every fork open; the brand, packs and community stay here |
| Big players add local AI | they will not go free and open; stay ahead on openness, privacy and automation |
| One developer, a large roadmap | ship each phase on its own; contributors take self-contained pieces |
| Support load | GitHub Discussions, Discord community, docs |
