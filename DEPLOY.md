# Deploying to Hostinger

A step-by-step guide to take this repo from local files to a live `houseofideas.in` (or whichever domain you chose) on Hostinger shared hosting.

## What you need before you start

- A Hostinger account with at least the "Premium" or "Business" web hosting plan (any plan that gives you `public_html` access)
- A domain (registered with Hostinger or pointed at Hostinger nameservers from elsewhere)
- This repo, on your local machine

You do **not** need Node.js, npm, or any build tools. This is a no-build static site.

---

## 1. Final pre-flight check (5 min)

Before uploading, sweep these files for placeholders you need to replace.

| File | What to replace |
|---|---|
| `apply.html` | Calendly URL: search for `https://calendly.com/your-handle/fellowship-intro` and swap in your real Calendly link |
| `index.html`, `studio.html`, `fellowship.html`, `partners.html`, `ideas.html`, `insights.html`, `contact.html`, `404.html`, `privacy.html`, `terms.html` | LinkedIn / X URLs in the footer (currently `href="#"`) |
| `index.html` etc. | `hello@houseofideas.in` &mdash; replace if your real email differs |
| `sitemap.xml` and all `<link rel="canonical">` and `<meta property="og:url">` tags | Replace `houseofideas.in` with your actual domain if different |
| `assets/img/favicon.svg` | Replace with your real favicon if/when you commission one |

Optional but recommended: add a real Open Graph image at `assets/img/og.jpg` (1200&times;630) and reference it from every page&rsquo;s `<head>` with `<meta property="og:image">`.

---

## 2. Set up the domain (10 min)

### If your domain is already with Hostinger
Skip to step 3.

### If your domain is registered elsewhere
Log in to your registrar, find DNS / nameserver settings, and point the domain to Hostinger&rsquo;s nameservers:

```
ns1.dns-parking.com
ns2.dns-parking.com
```

(Confirm the current nameservers in Hostinger&rsquo;s hPanel &rarr; *Domains* &rarr; the domain &rarr; *Details*.)

DNS propagation can take from a few minutes to a few hours.

---

## 3. Upload the site (10 min)

### Option A &mdash; via Hostinger File Manager (easiest)

1. In hPanel, go to *Files* &rarr; *File Manager*.
2. Navigate into `public_html/`. Delete the default `default.php` or `index.html` placeholder Hostinger drops in.
3. On your local machine, zip the contents of this repo &mdash; **not the parent folder**. From the repo root, select:
   - `index.html`, `studio.html`, `fellowship.html`, `ideas.html`, `partners.html`, `apply.html`, `insights.html`, `contact.html`, `404.html`, `privacy.html`, `terms.html`, `foundations.html`
   - `assets/` (whole folder)
   - `.htaccess`
   - `robots.txt`
   - `sitemap.xml`

   **Do not include:** `.git/`, `docs/`, `agent/`, `brand/`, `partials/`, `DEPLOY.md`, `README.md` (these are internal &mdash; not for the live site).

4. Right-click in File Manager &rarr; *Upload* &rarr; upload the zip.
5. Right-click the uploaded zip &rarr; *Extract* &rarr; extract into `public_html/`.
6. Delete the zip after extraction.

### Option B &mdash; via SFTP (faster for repeat deploys)

1. In hPanel, go to *Files* &rarr; *FTP Accounts*. Note your SFTP host, username, and password.
2. Use any SFTP client (FileZilla, Cyberduck, Transmit, or `scp` from the command line).
3. Connect, navigate to `public_html/` on the remote, and drag the same files listed above into it.

**Do not upload:** `.git/`, `docs/`, `agent/`, `brand/`, `partials/`, deployment guides.

---

## 4. Verify the deploy (5 min)

Visit:

- `https://houseofideas.in/` &mdash; home should load
- `https://houseofideas.in/studio` &mdash; clean URL should work (`.htaccess` strips `.html`)
- `https://houseofideas.in/this-doesnt-exist` &mdash; should land on the 404 page
- `https://houseofideas.in/sitemap.xml` &mdash; should render the sitemap

If the clean URLs (e.g. `/studio`) don&rsquo;t work, your Hostinger plan&rsquo;s Apache may not have `mod_rewrite` enabled, or the `.htaccess` file didn&rsquo;t upload. Re-check the File Manager and confirm `.htaccess` exists in `public_html/` (it&rsquo;s a hidden file &mdash; toggle &ldquo;Show hidden files&rdquo; in File Manager).

---

## 5. SSL / HTTPS (auto, 10&ndash;30 min)

Hostinger auto-provisions Let&rsquo;s Encrypt SSL on linked domains. In hPanel:

1. Go to *Advanced* &rarr; *SSL*.
2. Confirm the certificate is *Active* for your domain.
3. Enable *Force HTTPS* if available (the `.htaccess` already does this, but the panel toggle is a belt-and-braces).

If SSL hasn&rsquo;t provisioned within an hour, contact Hostinger support &mdash; sometimes it&rsquo;s a one-click fix on their end.

---

## 6. Analytics (Plausible recommended)

1. Sign up at [plausible.io](https://plausible.io) (~$9/mo for hobby tier &mdash; cookie-free, no banner needed, ~1KB).
2. Add your domain.
3. Plausible gives you a snippet like:

   ```html
   <script defer data-domain="houseofideas.in" src="https://plausible.io/js/script.js"></script>
   ```

4. Paste this snippet inside the `<head>` of **every page** (index, studio, fellowship, ideas, partners, apply, insights, contact, 404, privacy, terms).
5. Re-upload the modified files.

If you prefer Google Analytics 4 instead: same process, swap the snippet. (GA4 requires a cookie banner under GDPR &mdash; one more thing to maintain. Plausible avoids this.)

---

## 7. Search Console

1. Sign in to [Google Search Console](https://search.google.com/search-console).
2. Add your property (`https://houseofideas.in`).
3. Verify ownership &mdash; the easiest method is the DNS TXT record (set this in Hostinger&rsquo;s DNS panel) or the HTML meta tag method.
4. Once verified, submit your sitemap: `https://houseofideas.in/sitemap.xml`.

Same process for Bing Webmaster Tools, if you care about Bing.

---

## 8. Post-launch checklist

- [ ] All pages load on desktop and mobile
- [ ] Mobile nav opens/closes correctly
- [ ] All footer links resolve
- [ ] `/apply` Calendly button opens your real Calendly URL
- [ ] `mailto:` links open your mail client with the correct subject lines
- [ ] LinkedIn / X links in nav and footer point to your real handles
- [ ] Favicon shows in the browser tab
- [ ] OG preview renders correctly when the URL is shared on LinkedIn / X (use [opengraph.xyz](https://www.opengraph.xyz/) to test)
- [ ] No broken links (run a quick crawl via [W3C link checker](https://validator.w3.org/checklink) or `httrack`)
- [ ] Lighthouse audit: Performance &ge; 90, Accessibility &ge; 95, SEO &ge; 95
- [ ] Plausible is recording page views

---

## Updating the site after launch

1. Edit the file(s) on your local machine.
2. Upload only the changed files via File Manager or SFTP.
3. Hard-refresh your browser (Ctrl+Shift+R / Cmd+Shift+R) &mdash; the `.htaccess` sets a short 10-min cache on HTML, so changes propagate quickly.

For CSS / JS changes, you may need to append a cache-busting query string to the `<link>` and `<script>` tags during heavy iteration (e.g. `main.css?v=2`). Not necessary for routine edits.

---

## Things to revisit before scale

- Move from Google Fonts (CDN) to self-hosted fonts &mdash; saves a third-party request and tightens privacy. Drop the `.woff2` files into `assets/fonts/` and `@font-face` them in `tokens.css`.
- Generate a full favicon set (16, 32, 96, 180, 192, 512 px PNGs + apple-touch-icon) using [realfavicongenerator.net](https://realfavicongenerator.net/) and replace the single SVG.
- Commission a real OG image (1200&times;630) for social sharing.
- Replace the privacy.html / terms.html drafts with counsel-reviewed versions before the site goes public.
- If the page list grows beyond 12&ndash;15, migrate to a tiny static-site generator (11ty, Astro) so nav/footer aren&rsquo;t duplicated. Half a day&rsquo;s work; reversible.

---

*This guide is intentionally specific. If a step looks wrong against your version of Hostinger&rsquo;s hPanel (they change UI occasionally), the underlying intent should still hold.*
