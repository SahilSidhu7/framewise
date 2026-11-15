# AIEditor: how it earns and how it gets users

Positioning: **"The AI video editor that runs on your PC. Private. No subscription. Automate it."**

## 1. Market in short
- Video editing software is roughly a $3.5–4.7B market in 2026 (third-party estimates disagree), growing about 6% a year. CapCut has 300M+ monthly users.
- A do-everything editor cannot beat CapCut, Adobe or free DaVinci Resolve head-on.
- Open gaps:
  - **Local, private, no subscription, with good AI.** Open-source rivals (OpenCut, Kdenlive, Shotcut) have weak or no AI; CapCut raised its Pro price and its terms are widely disliked.
  - **Automation.** No mainstream editor has a local API. Cloud clipping tools (OpusClip and similar) charge per minute of video.
  - **Kid-friendly VFX.** "Make me shoot lightning" has no good desktop answer.
- Decision: lead with the automation engine (buyers with budgets, least competition), then the movie timeline, kid mode and VFX.

## 2. Tiers
Rule: **no server of ours runs AI unless a customer has already paid for that exact job.**

| Tier | What | Price idea | Our cost |
|---|---|---|---|
| Free | Full local editor: photo + video, AI prompts with style/output options, captions, export with no watermark | 0 | 0 (runs on their PC) |
| Pro licence | Automation: headless mode, public API, webhooks, CLI, watch folders, batch, n8n node unlock, several reels per video, premium style packs | one-time about $59, or $8/month | 0 (offline-signed licence key, checked locally) |
| Studio / agency | Commercial multi-seat, priority support | about $199/year per seat | 0 |
| Packs | VFX packs, style packs, music/SFX packs; later a marketplace with revenue share | $5–15 each | 0 (file download) |
| Connect | Reach your own PC's engine from a phone or from cloud n8n/Zapier through a secure tunnel. The PC does all the AI work | about $4/month | near 0 (Cloudflare Tunnel / Workers free tier; the relay carries small control messages only) |
| Cloud render credits | For people with weak PCs: the job runs on a rented GPU | prepaid credits at 3–4x GPU cost | pay-per-second, scale-to-zero serverless GPU; no job, no bill |

Prices are starting points to test, not commitments.

## 3. Cloud without paying for cloud
In the order to use them:
1. **Local first.** The free tier costs us nothing.
2. **The user's PC is the cloud** (Connect).
3. **Bring your own.** The user pastes their own API key, or launches our one-click template on their own GPU-rental account and pays that provider directly.
4. **Prepaid serverless** for cloud render: credits are bought before use, so revenue arrives before cost. Never an always-on GPU server.
5. **Free credits** for development and beta: startup programmes from NVIDIA, Microsoft, Google and AWS, plus serverless providers' free monthly credit. Check current terms when applying.
6. **Own hardware for beta only:** the existing home server can host the relay, the licence-key generator, the landing page and a small beta queue.

Unavoidable costs: a domain (about $12/year); Windows code signing (about $10/month, or a free programme for open-source projects); payment fees per sale. Installer and model downloads go through GitHub Releases and Hugging Face at no cost.

## 4. Payments and licences
- A merchant of record (Lemon Squeezy, Paddle or Gumroad) takes payment, handles worldwide sales tax and pays out. No monthly fee, roughly 5% plus fees per sale, built-in licence keys and affiliate links. Works from India without a foreign company.
- Licence keys are signed offline (Ed25519) and verified inside the app: no licence server to run, and the app keeps working offline.
- **Open core (recommended, to confirm before release):** publish the free editor's source under AGPL so nobody can repackage it closed; keep the Pro automation modules and packs closed. Open source is what makes "private, runs locally" believable.

## 5. Marketing (budget: time, not money)
1. **Dogfood content.** Every demo, reel and short is made with the tool, with the prompt on screen. Before/after clips are the ad.
2. **Automation crowd first.** Publish n8n workflow templates ("YouTube video in, 5 captioned shorts out, posted automatically"), list the community node, post in r/n8n, r/selfhosted, r/LocalLLaMA and the n8n forum; short tutorials; a 30% affiliate link for automation YouTubers.
3. **Launch moments.** GitHub repo with a strong README and GIFs, Show HN, Product Hunt, AlternativeTo, and comparison pages for "CapCut alternative" and "OpusClip alternative" searches.
4. **Waitlist** landing page before release (free static hosting), an email list, a Discord for beta users; 50 hand-picked beta users from the n8n community.
5. **Kid / VFX wave** with phase F: "my 10-year-old made this" superpower clips on Shorts, TikTok and Reels; parent and homeschool groups; a free classroom licence for teachers.
6. **Measure** with opt-in, privacy-respecting counts only: downloads, activations, Pro conversion. First 3 months after release: aim for 2,000 downloads and 3% Pro.

## 6. Risks
| Risk | Answer |
|---|---|
| Needs a decent PC; casual users edit on phones | Connect (phone as remote for the PC), cloud render credits, a Lite hardware tier |
| Big players add local AI or an API | stay ahead on automation depth and privacy; open core builds trust they can't copy |
| Kids market is slow (parents pay, schools buy slowly) | it comes last and rides on the VFX hook; revenue does not depend on it |
| One developer, about 33 weeks of roadmap | first sellable release after about 12 weeks; each phase ships on its own |
| Support load | Discord community, docs, the Studio tier pays for priority support |
