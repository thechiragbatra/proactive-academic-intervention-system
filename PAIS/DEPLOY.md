# Deploying PAIS to Render.com

A step-by-step guide to getting PAIS live on the public internet using
Render's free tier. **Total time: ~10 minutes.** No credit card required.

---

## What you'll end up with

A publicly-accessible URL like `https://pais-yourname.onrender.com` that
serves the full Flask app — dashboard, student detail, edit page with live
risk preview, CSV upload, the works.

**Free-tier caveats** (not blockers, just good to know):

- The service **sleeps after 15 minutes of inactivity**. First request after
  sleep takes ~30 seconds to wake it up + ~15 seconds for pipeline warmup.
  After that it's snappy. *Warm it up 2 minutes before your viva.*
- **Disk is ephemeral.** Edits saved via the edit page persist in memory
  while the service runs, but are wiped on each redeploy/restart. See the
  "Edit persistence" section below for the fix.
- **512 MB RAM, single CPU.** More than enough for PAIS (it uses ~200 MB).

---

## Prerequisites

You need three accounts, all free:

1. **GitHub account** — to host your code. Sign up at https://github.com if
   you don't have one.
2. **Git installed locally** — check with `git --version`. If missing:
   - Windows: https://git-scm.com/download/win
   - Mac: `brew install git` or already installed
   - Linux: `sudo apt install git`
3. **Render account** — sign up at https://render.com (use "Sign in with
   GitHub" — it's the smoothest path).

---

## Step 1 — Push the project to GitHub

Open a terminal in the `PAIS` folder and run:

```bash
# Initialize the repository
git init
git branch -M main

# Tell git who you are (one-time setup)
git config user.name "Your Name"
git config user.email "your_email@example.com"

# Stage everything respecting .gitignore
git add .
git commit -m "Initial commit: PAIS v1.0"
```

Now create the remote repo on GitHub:

1. Go to https://github.com/new
2. Repository name: `pais` (or whatever you like)
3. Set it to **Public** (Render's free tier requires public repos to
   auto-deploy) — or Private if you upgrade later
4. **Do NOT** tick "Add a README", "Add .gitignore", or "Choose a license" —
   the files already exist locally
5. Click **Create repository**

GitHub shows you a set of commands. Run the `push` section:

```bash
git remote add origin https://github.com/YOUR_USERNAME/pais.git
git push -u origin main
```

You'll be prompted for your GitHub username and a **Personal Access Token**
(not your password — GitHub removed password auth in 2021). If you don't
have a token yet:

1. https://github.com/settings/tokens → *Generate new token (classic)*
2. Scope: tick `repo`
3. Copy the token (you only see it once)
4. Use it as the password when git prompts

After a successful push, refresh the GitHub repo page — you should see all
your files there.

---

## Step 2 — Deploy on Render (blueprint method, recommended)

The repo contains `render.yaml`, which is a Blueprint that tells Render
exactly how to build and run the app. You just point Render at the repo.

1. Go to https://dashboard.render.com/
2. Click **New +** (top-right) → **Blueprint**
3. Click **Connect GitHub** if this is your first time; authorize Render
4. Search for and select your `pais` repository
5. Click **Connect**
6. Render shows a summary of what it'll create:
   - one **Web Service** named `pais`
   - plan: **Free**
   - region: **Singapore**
7. Click **Apply**

That's it. Render starts the build.

### What you'll see during build (~5 minutes):

1. *Cloning repository...* — pulls your code from GitHub
2. *Running build command:* `pip install ... && python main.py` — installs
   Python packages and runs the full pipeline. You'll see log lines like:
   - `Processed 5,000 rows → /.../students_processed.csv`
   - `training logistic_regression...`
   - `Winner: logistic_regression`
3. *Starting service with gunicorn...*
4. *Your service is live 🎉*
5. Render displays your URL at the top of the page.

Click the URL. The **first load takes ~15 seconds** because the pipeline is
warming up. Subsequent requests are fast.

---

## Alternative — manual deploy (if blueprint fails)

If for any reason the blueprint approach errors out, you can configure
things by hand:

1. Render dashboard → **New +** → **Web Service**
2. Connect your GitHub repo
3. Fill in:
   - **Name:** `pais` (or anything)
   - **Region:** Singapore (or your closest)
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt && python main.py --skip-train || python main.py
     ```
   - **Start Command:**
     ```
     gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
     ```
   - **Plan:** Free
4. (Optional) Environment tab:
   - `PYTHON_VERSION` = `3.11.9`
   - `FLASK_ENV` = `production`
   - `PYTHONUNBUFFERED` = `1`
5. Click **Create Web Service**

---

## Step 3 — Verify it works

Once the service is live:

- [ ] Dashboard (`/`) shows 5,000 students, real KPIs, charts render
- [ ] Students (`/students`) — search works, filter works
- [ ] Click any student → detail page with 84-day chart
- [ ] Click **Edit details** → form loads with live risk preview
- [ ] Drag a slider on the edit page → risk score updates in real-time
      (this confirms the `/api/student/<id>/simulate` endpoint is working)
- [ ] Model page (`/model`) shows F1, AUC, confusion matrix image

If any of these fail, check the Render logs (service page → **Logs** tab).

---

## Redeploying after changes

Every push to `main` triggers an automatic redeploy (this is what the
`autoDeploy: true` line in `render.yaml` does). Workflow:

```bash
# edit some files locally
git add .
git commit -m "Improve student detail page"
git push
# Render starts rebuilding automatically — ~5 min later you're live
```

You can watch the build in Render's dashboard → service → **Events** tab.

---

## Keeping the app warm for your viva

Render free tier spins down after 15 minutes of inactivity. On the morning
of your viva:

1. **2 hours before:** push a trivial commit (or click "Manual Deploy" in
   Render dashboard) to force a fresh build.
2. **10 minutes before:** open the dashboard URL in your browser. Wait for
   the first load (~30 seconds). Service is now warm.
3. **During viva:** navigate as normal. As long as you don't idle for 15+
   minutes, it stays hot.

**Alternative: a free cron-ping service.** https://cron-job.org (free) can
hit your URL every 14 minutes to keep it from sleeping. Set up one job
pointing at your dashboard URL.

---

## Edit persistence on Render free tier

The `reports/edits_overlay.jsonl` file keeps edits alive across app
restarts, BUT not across redeploys — each redeploy gets a fresh filesystem.
For a viva demo this is actually fine: edits persist for the duration of
your demo session.

**If you want edits to genuinely persist**, upgrade to Render's paid tier
($7/mo) and add a persistent disk:

```yaml
# add to render.yaml under the web service
    disk:
      name: pais-data
      mountPath: /opt/render/project/src/reports
      sizeGB: 1
```

This mounts a 1 GB disk at `reports/` so edits and notifications survive
restarts and redeploys.

---

## Common deployment issues

**Build fails with `ModuleNotFoundError: No module named 'gunicorn'`**
The build ran out of memory on pip install — Render free has a 512 MB build
limit. Try upgrading to the Starter plan ($7/mo) or remove `matplotlib` from
`requirements.txt` (the figures are already checked in).

**First page loads blank / hangs for >60 seconds**
The free tier is warming up from sleep. Refresh the page once — the second
load should be fast.

**"Application exited early" in Render logs**
Check the deploy logs for the actual error. Most common cause: the model
file or processed data wasn't generated during build. Fix by editing the
build command to force a full pipeline run: remove the `--skip-train ||`
portion.

**502 Bad Gateway on `/` but other routes work**
Health check probably timed out. In the Render dashboard → service →
Settings, raise *Health Check Timeout* to 60 seconds. Or change the health
check path to `/api/stats` which is lighter than the full dashboard render.

**CSV upload returns 500**
Render's free tier has an ephemeral filesystem. Uploads land in
`webapp/uploads/` which works during the session but resets on redeploy.
For production upload persistence, mount a disk (see above).

---

## Removing or pausing the deployment

From Render dashboard → service → **Settings** → **Suspend Service**
(pauses billing if on paid tier) or **Delete Service** (permanent).

Your GitHub repo stays intact either way.

---

## Costs summary

| Scenario | Monthly cost |
|---|---|
| Free tier, occasional demo use | **$0** |
| Free tier + cron-ping for always-warm | **$0** |
| Starter plan (no sleep, 0.5 CPU, 512 MB) | $7 |
| Starter + 1 GB persistent disk | $7 + $0.25/GB = ~$7.25 |

For a minor project demo, **free tier is the right call**. Put the URL on
your CV, keep the repo public as a portfolio piece, and you're done.
