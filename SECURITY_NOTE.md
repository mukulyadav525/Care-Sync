# Security note: db.sqlite3 was committed to git

## What happened

`database/db.sqlite3` was tracked in git (marked "intentional" in an old
`.gitignore` comment, from when it only held demo/test data). By the time of
this review it contained **21 real user accounts** — usernames, real email
addresses, and PBKDF2 password hashes — plus device keys, alert history, and
consent records. That data has been in 3 git commits.

This has been fixed going forward: the file is now `git rm --cached` and
`.gitignore`'d (see the Database section), so future commits won't include
it. **That does not remove it from git history** — anyone with a clone of
this repo (or access to wherever it's hosted) can still recover those old
commits until the history itself is rewritten.

## What you need to decide/do

1. **Check exposure.** Is this repo private on GitHub, or has it ever been
   public / shared with anyone outside your immediate team? If it's been
   private the whole time with a small, trusted collaborator list, the
   practical risk is low. If it was ever public, treat the emails and
   password hashes as disclosed.

2. **Rotate what's rotatable.**
   - `DJANGO_SECRET_KEY` — already environment-only (not in the DB or repo),
     but rotate it anyway on the server as routine hygiene; this invalidates
     all existing sessions/JWTs, which is a good side effect here.
   - Password hashes: PBKDF2 with 600k–1.2M iterations is strong, but if this
     was ever public, the safest move is forcing a password reset for the 21
     accounts (`python manage.py changepassword <username>` per account, or
     add a one-off migration that expires all passwords / require reset on
     next login).

3. **Purge it from git history** (only if you've confirmed real exposure —
   this rewrites history and requires every collaborator to re-clone):
   ```bash
   pip install git-filter-repo
   git filter-repo --path database/db.sqlite3 --invert-paths
   git push origin --force --all
   ```
   Coordinate with anyone else who has a clone — their local history will
   diverge and they'll need to re-clone rather than pull.

4. Going forward, only `database/.gitkeep` (or nothing) should be tracked in
   `database/`; the real `db.sqlite3` lives only on the server, created by
   `manage.py migrate` and backed up separately (e.g. a nightly `sqlite3
   .backup` cron job to encrypted storage), never committed.

This file is a one-time record of the incident and remediation steps — safe
to delete once you've acted on the items above.
