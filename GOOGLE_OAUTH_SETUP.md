# Google Sign-In Setup

Optional Google identity badge. The app — sources, chat, citations, evaluation —
works fully **signed out**: on load the frontend calls `/api/auth/local-login` and
runs as the built-in `local` account. Google sign-in only swaps that for a real
identity (name / email) shown as a chip in the top bar. When
`GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` are unset, `/api/auth/status` reports
`google_oauth: false`, the button stays hidden, and nothing breaks.

**No backend change is needed.** `ragchat/auth.py` already implements the full flow
with `itsdangerous` signed cookies (no `authlib`, no Starlette `SessionMiddleware`).
Only the frontend wiring (§4) and env vars (§2) are missing.

## 1. Create an OAuth 2.0 Client ID

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and select (or
   create) a project.
2. **APIs & Services → OAuth consent screen**
   - User type: **External** (use **Internal** for a Workspace-only app).
   - Fill in app name, support email, developer contact.
   - Scopes: the defaults are enough — `openid`, `email`, `profile`.
   - While in **Testing**, add your Google account under **Test users**, or
     **Publish app** to allow anyone.
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   - Application type: **Web application**
   - Name: e.g. `rag-gel web`
   - **Authorized redirect URIs** — add all three (note the `/google/` segment):
     ```
     https://rag-gel.vercel.app/api/auth/google/callback
     http://localhost:5173/api/auth/google/callback
     http://localhost:4173/api/auth/google/callback
     ```
     (5173 = `vite dev`, 4173 = `vite preview`; both proxy `/api` to FastAPI.)
4. Copy the **Client ID** and **Client secret**.

The redirect URI must match `GOOGLE_REDIRECT_URI` (§2) *character for character* —
`redirect_uri_mismatch` is almost always a typo here.

## 2. Environment variables

Set these in the **Vercel dashboard** (Project → Settings → Environment Variables) and
in a local `.env`. All are read via `os.environ` (`ragchat/config.py`).

| Variable | Required? | Value |
| --- | --- | --- |
| `GOOGLE_CLIENT_ID` | for Google sign-in | from step 1 |
| `GOOGLE_CLIENT_SECRET` | for Google sign-in | from step 1 |
| `GOOGLE_REDIRECT_URI` | for Google sign-in | the exact callback registered in step 1, e.g. `https://rag-gel.vercel.app/api/auth/google/callback` |
| `SESSION_SECRET` | production | long random string (below) |

`GOOGLE_REDIRECT_URI` is **not** derived from the request — if unset, `google_auth_url()`
sends an empty `redirect_uri` and Google rejects the consent request. Set it whenever
`GOOGLE_CLIENT_ID` is set.

Generate a `SESSION_SECRET`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Without it, `settings.session_secret` falls back to `"dev-session-secret"` — anyone who
knows that placeholder can forge a session cookie. Never ship it to production.

Redeploy after changing variables; Vercel injects them at build/boot time.

## 3. How it works (real endpoints)

All routes live in `ragchat/app.py`; logic in `ragchat/auth.py`. Session = signed
`ragchat_session` cookie (httponly), verified by `get_current_user`, which gates
`/api/documents`, `/api/chats`, etc.

| Route | Method | Purpose |
| --- | --- | --- |
| `/api/auth/status` | GET | `{"authenticated": bool, "user": {"id","name","email"}\|null, "google_oauth": bool}` — drives the badge |
| `/api/auth/google/login` | GET | Builds the consent URL (`scope=openid email profile`, `prompt=select_account`), sets an `oauth_state` cookie, redirects to Google |
| `/api/auth/google/callback` | GET | Verifies `state`, exchanges the code via `httpx`, `find_or_create_google_user`, sets `ragchat_session`, redirects to `/` |
| `/api/auth/local-login` | POST | Signs in as the built-in `local` account (default; keeps the app usable signed out) |
| `/api/auth/login` | POST | Username/password fallback, sets `ragchat_session` |
| `/api/auth/logout` | POST | Clears `ragchat_session` |

There is **no** `/api/auth/me`, no `GET /api/auth/login`, and no `GET /api/auth/callback`.

## 4. Frontend wiring fixes (`frontend/app.js`)

Three edits. Copy-paste ready.

**a. Google button target — use the real login route** (~line 128):
```js
// old
window.location.href = "/api/auth/login";
// new
window.location.href = "/api/auth/google/login";
```

**b. Status fetch — hit the real endpoint** (~line 181):
```js
// old
const res = await fetch("/api/auth/me", { credentials: "same-origin" });
// new
const res = await fetch("/api/auth/status", { credentials: "same-origin" });
```

**c. Consume the `/api/auth/status` shape** — replace the `me`-shape branching at the
end of `initGoogleAuth()` (~lines 186–187) and the render helpers, which currently read
the old `/api/auth/me` fields (`logged_in`, `configured`, `picture`):
```js
// old
if (me && me.logged_in) renderSignedIn(slot, me);
else renderSignedOut(slot, me ? me.configured : undefined);
// new
if (me && me.authenticated && me.user) renderSignedIn(slot, me.user);
else renderSignedOut(slot, me ? me.google_oauth : undefined);
```
- `renderSignedIn(slot, user)` now takes the `user` object → read `user.name` / `user.email`.
  The status payload carries no `picture`, so the chip falls back to the initial avatar.
- `renderSignedOut(slot, configured)` keeps its signature; `configured` is now
  `me.google_oauth`. The **button renders only when `google_oauth === true`** — otherwise
  keep the slot empty so no broken `/api/auth/google/login` call is possible.
- Sign-in stays **optional and non-gating**: `/api/auth/local-login` on load keeps the
  app fully working whether or not Google is configured.

## 5. Verification

- **Local, Google configured** (`GOOGLE_CLIENT_ID`/`SECRET`/`REDIRECT_URI` set): `npm run dev`,
  then Playwright-click "Sign in with Google" → expect a redirect to Google consent, a
  `ragchat_session` cookie on return, and the name chip in the top bar.
- **Local, Google unset**: reload → button hidden, `local-login` auto sign-in, every
  feature usable.
- **Live**: repeat on `https://rag-gel.vercel.app` after the env vars are set and redeployed.

## 6. Vercel deployment protection

Keep **production deployment protection OFF** (Project → Settings → Deployment Protection
→ Vercel Authentication: *Disabled* for Production). Leaving it on stacks a Vercel SSO
login in front of the site (two logins) and blocks Google's redirect back to
`/api/auth/google/callback` for anyone outside your Vercel team.

Preview protection is fine to leave on if you want private previews — just register that
preview URL's `/api/auth/google/callback` as an authorized redirect URI too.

## Note on better-auth

better-auth is a Node-only library (needs a Node runtime + JS DB adapter). This backend
is Python/FastAPI and signs its own `ragchat_session` cookies with `itsdangerous` in
`ragchat/auth.py` — **no `authlib` dependency is required**. better-auth would only make
sense if the app were rehosted on a Node framework (e.g. Next.js).
