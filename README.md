# Beat Butler — hosted test build

This repository holds the pieces needed for a hosted test of Beat Butler:

- `services/analyze/` — FastAPI + librosa worker (Docker image built from its Dockerfile)
- `web/` — the built public web bundle, served as a static site

The worker receives uploaded audio for analysis and deletes its temp file immediately. No audio is stored. The web app keeps the original file in the browser's IndexedDB only when the producer explicitly opts in, so a Release Packet can be exported locally. Release packets are downloaded by the producer; nothing is submitted to a DSP or PRO.

Supabase accounts are disabled in this test build, so the library is saved per browser in `localStorage`.
