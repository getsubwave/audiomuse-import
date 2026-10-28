#!/usr/bin/env node
// Import pre-computed analysis from an AudioMuse-AI instance into SUB/WAVE's
// library.db, skipping the slow LLM tagging + BPM/key pass. Standalone: talks
// only to AudioMuse (GET /api/sync) and SUB/WAVE's state/library.db. AudioMuse
// keys by the media-server track id, which IS tracks.id, so the join needs no
// Navidrome auth.
//
// Imports:  tempo -> bpm, key+scale -> Camelot musical_key, energy ->
//           low/medium/high, mood_vector/other_features -> moods, top genre tag.
// Skips:    CLAP/MusiCNN embeddings, and leaves analysis_version NULL so
//           SUB/WAVE's acoustic analysis still adds outro/structure/embeddings.

import Database from 'better-sqlite3';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { mapTrack } from './map.mjs';

// Column-set version this tool writes against; matches TAGGER_VERSION in
// SUB/WAVE's controller/src/music/library-db/handle.ts
// (https://github.com/perminder-klair/subwave), so imported moods count as
// current tags. The schema itself is owned there; this tool never migrates.
// MIN_USER_VERSION is the earliest schema carrying every column written here
// (v16 moved genre storage to the `genres` JSON array and made `genre` a
// GENERATED column, which rejects writes); below it, refuse and tell the user
// to boot the controller once to migrate.
const TAGGER_VERSION = 3;
const MIN_USER_VERSION = 16;

function parseArgs(argv) {
  const args = { concurrency: 8, moodCutoff: 0.4 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--audiomuse-url') args.audiomuseUrl = next();
    else if (a === '--library-db') args.libraryDb = next();
    else if (a === '--concurrency') args.concurrency = Math.max(1, Number(next()) || 8);
    else if (a === '--limit') args.limit = Number(next()) || undefined;
    else if (a === '--mood-cutoff') {
      // Guard NaN/out-of-range: an unparseable value would silently disable
      // mood filtering, since `score < NaN` is always false
      // (perminder-klair/subwave#934).
      const mc = Number(next());
      args.moodCutoff = Number.isFinite(mc) ? Math.min(1, Math.max(0, mc)) : 0.4;
    }
    else if (a === '--apikey') args.apikey = next();
    else if (a === '--dry-run') args.dryRun = true;
    else if (a === '--overwrite') args.overwrite = true;
    else if (a === '-h' || a === '--help') args.help = true;
    else {
      console.error(`Unknown argument: ${a}`);
      args.badArg = true;
    }
  }
  return args;
}

const HELP = `audiomuse-import — import AudioMuse-AI analysis into SUB/WAVE's library.db

Usage:
  audiomuse-import --audiomuse-url http://audiomuse:8000 \\
    --library-db /path/to/subwave/state/library.db [options]

Options:
  --audiomuse-url <url>   AudioMuse base URL (or AUDIOMUSE_URL env). Required.
  --apikey <token>        Bearer token for the AudioMuse API (or
                          AUDIOMUSE_API_KEY env). Sent as
                          "Authorization: Bearer <token>".
  --library-db <path>     Path to SUB/WAVE's library.db (default:
                          $STATE_DIR/library.db, else ./state/library.db).
  --overwrite             Overwrite existing bpm/moods/key/energy (default:
                          fill only empty fields, never clobber SUB/WAVE's own).
  --dry-run               Map and report (incl. how many tracks would update
                          vs be added), write nothing.
  --concurrency <n>       Reserved; paging is currently sequential.
  --mood-cutoff <0..1>    Min tag score to count a mood (default 0.4).
  --limit <n>             Stop after N AudioMuse tracks (testing).
  -h, --help              This help.

Docs: https://github.com/getsubwave/audiomuse-import
`;

async function* iterateAudioMuseTracks(baseUrl, { limit, apikey } = {}) {
  const root = baseUrl.replace(/\/+$/, '');
  const headers = {};
  if (apikey) headers.Authorization = `Bearer ${apikey}`;
  let page = 1;
  let seen = 0;
  let providerChecked = false;
  while (true) {
    const url = `${root}/api/sync?page=${page}&limit=500&include_embeddings=false`;
    const res = await fetch(url, { headers });
    if (!res.ok) {
      throw new Error(`GET ${url} -> ${res.status} ${res.statusText}`);
    }
    // A proxy/login page can answer 200 with HTML, and a wrong endpoint can
    // return JSON with no `tracks` array; parse defensively so either gives a
    // clear message rather than a SyntaxError or a silent zero-track success.
    let body;
    try {
      body = await res.json();
    } catch {
      throw new Error(
        `GET ${url} returned ${res.status} but the body was not JSON — is ${root} ` +
          `an AudioMuse API? (a proxy or login page can answer 200 with HTML.)`,
      );
    }
    if (!body || !Array.isArray(body.tracks)) {
      const shape = body && typeof body === 'object'
        ? `keys: ${Object.keys(body).join(', ') || 'none'}`
        : `type: ${typeof body}`;
      throw new Error(
        `GET ${url} returned JSON with no "tracks" array (${shape}). ` +
          `Check the AudioMuse URL and that /api/sync is available.`,
      );
    }
    if (!providerChecked) {
      providerChecked = true;
      if (body.provider_type && body.provider_type.toLowerCase() !== 'navidrome') {
        throw new Error(
          `AudioMuse is configured for '${body.provider_type}', not navidrome. ` +
            `Its item ids won't match SUB/WAVE's Subsonic ids, so the import can't join. ` +
            `Point AudioMuse at the same Navidrome SUB/WAVE uses.`,
        );
      }
    }
    const tracks = body.tracks;
    for (const t of tracks) {
      if (limit && seen >= limit) return;
      seen++;
      yield t;
    }
    if (!body.has_more || (limit && seen >= limit)) return;
    page = body.next_page || page + 1;
  }
}

function makeWriter(db) {
  const selectStmt = db.prepare(
    `SELECT bpm, musical_key, moods FROM tracks WHERE id = ?`,
  );
  const insertStmt = db.prepare(
    `INSERT INTO tracks (id, title, artist, album, year, genres, bpm, musical_key,
        moods, energy, source, tagger_version, tagged_at)
     VALUES (@id, @title, @artist, @album, @year, @genres, @bpm, @musicalKey,
        @moods, @energy, 'manual', @taggerVersion, @now)`,
  );

  return function writeTrack(id, meta, mapped, { overwrite }) {
    const existing = selectStmt.get(id);
    const now = new Date().toISOString();
    const moodsJson = mapped.moods.length ? JSON.stringify(mapped.moods) : null;

    if (!existing) {
      insertStmt.run({
        id,
        title: meta.title ?? null,
        artist: meta.artist ?? null,
        album: meta.album ?? null,
        year: Number.isFinite(meta.year) ? meta.year : null,
        genres: mapped.genre != null ? JSON.stringify([mapped.genre]) : null,
        bpm: mapped.bpm,
        musicalKey: mapped.musicalKey,
        moods: moodsJson,
        energy: mapped.energy,
        taggerVersion: moodsJson ? TAGGER_VERSION : null,
        now,
      });
      return moodsJson || mapped.bpm != null ? 'inserted' : 'inserted-empty';
    }

    // Existing row: fill gaps unless --overwrite. Build a targeted UPDATE.
    const sets = [];
    const params = { id };
    const wantBpm = overwrite || existing.bpm == null;
    const wantKey = overwrite || existing.musical_key == null;
    const wantMoods = overwrite || existing.moods == null;

    if (wantBpm && mapped.bpm != null) {
      sets.push('bpm = @bpm');
      params.bpm = mapped.bpm;
    }
    if (wantKey && mapped.musicalKey != null) {
      sets.push('musical_key = @musicalKey');
      params.musicalKey = mapped.musicalKey;
    }
    if (wantMoods && moodsJson) {
      sets.push('moods = @moods', "source = 'manual'",
        'tagger_version = @taggerVersion', 'tagged_at = @now');
      params.moods = moodsJson;
      params.taggerVersion = TAGGER_VERSION;
      params.now = now;
      // Energy rides with the mood write, but only when AudioMuse gave one: the
      // existence SELECT doesn't read energy, so a null here would clobber an
      // energy SUB/WAVE already computed (perminder-klair/subwave#934).
      if (mapped.energy != null) {
        sets.push('energy = @energy');
        params.energy = mapped.energy;
      }
    }
    // Genre only fills a truly empty genre list. Write `genres`, never `genre`:
    // the scalar is GENERATED from genres[0] and SQLite rejects writes to it.
    if (mapped.genre != null) {
      sets.push("genres = CASE WHEN genres IS NULL OR genres = '[]' THEN @genres ELSE genres END");
      params.genres = JSON.stringify([mapped.genre]);
    }

    if (!sets.length) return 'skipped';
    db.prepare(`UPDATE tracks SET ${sets.join(', ')} WHERE id = @id`).run(params);
    return 'updated';
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.badArg) {
    console.log(HELP);
    process.exit(args.badArg ? 2 : 0);
  }

  const audiomuseUrl = args.audiomuseUrl || process.env.AUDIOMUSE_URL;
  if (!audiomuseUrl) {
    console.error('Missing AudioMuse URL. Pass --audiomuse-url or set AUDIOMUSE_URL.\n');
    console.log(HELP);
    process.exit(1);
  }

  const apikey = args.apikey || process.env.AUDIOMUSE_API_KEY;
  const stateDir = process.env.STATE_DIR || resolve(process.cwd(), 'state');
  const dbPath = args.libraryDb || resolve(stateDir, 'library.db');
  if (!existsSync(dbPath)) {
    console.error(
      `library.db not found at ${dbPath}. Pass --library-db with the path to ` +
        `library.db inside your SUB/WAVE state/ folder. It is created the first ` +
        `time the station starts.`,
    );
    process.exit(1);
  }

  const db = new Database(dbPath);
  db.pragma('busy_timeout = 5000');
  const userVersion = db.pragma('user_version', { simple: true }) || 0;
  if (userVersion < MIN_USER_VERSION) {
    console.error(
      `library.db schema is v${userVersion}, need >= v${MIN_USER_VERSION}. ` +
        `Update SUB/WAVE and start it once so it upgrades the database, then re-run.`,
    );
    db.close();
    process.exit(1);
  }

  console.log(`AudioMuse: ${audiomuseUrl}`);
  console.log(`library.db: ${dbPath} (schema v${userVersion})`);
  console.log(args.overwrite ? 'Mode: OVERWRITE existing fields' : 'Mode: fill gaps only');
  if (args.dryRun) console.log('Mode: DRY RUN (no writes)');
  console.log('');

  const writeTrack = makeWriter(db);
  const existsStmt = db.prepare('SELECT 1 FROM tracks WHERE id = ?');
  const counts = { seen: 0, inserted: 0, updated: 0, skipped: 0, noSignal: 0 };
  const samples = [];

  const commit = db.transaction((batch) => {
    for (const t of batch) {
      const mapped = mapTrack(t, { moodCutoff: args.moodCutoff });
      const hasSignal = mapped.bpm != null || mapped.moods.length || mapped.musicalKey != null;
      if (!hasSignal) {
        counts.noSignal++;
        continue;
      }
      if (samples.length < 5) {
        samples.push({ title: t.title, artist: t.author, ...mapped });
      }
      if (args.dryRun) {
        // The one thing a dry run can't otherwise show: whether AudioMuse's ids
        // line up with this library. Mostly "added" on an established station
        // means they don't.
        if (existsStmt.get(t.id)) counts.updated++;
        else counts.inserted++;
        continue;
      }
      const outcome = writeTrack(t.id, { title: t.title, artist: t.author, album: t.album, year: t.year }, mapped, { overwrite: args.overwrite });
      if (outcome === 'inserted' || outcome === 'inserted-empty') counts.inserted++;
      else if (outcome === 'updated') counts.updated++;
      else counts.skipped++;
    }
  });

  // Buffer pages and commit in batches for a fast single transaction per chunk.
  let batch = [];
  const BATCH = 500;
  for await (const t of iterateAudioMuseTracks(audiomuseUrl, { limit: args.limit, apikey })) {
    counts.seen++;
    batch.push(t);
    if (batch.length >= BATCH) {
      commit(batch);
      batch = [];
      process.stdout.write(`\r  processed ${counts.seen} tracks…`);
    }
  }
  if (batch.length) commit(batch);
  process.stdout.write('\r');

  db.close();

  console.log('Sample mappings:');
  for (const s of samples) {
    console.log(
      `  ${s.artist ?? '?'} — ${s.title ?? '?'}: bpm=${s.bpm ?? '—'} key=${s.musicalKey ?? '—'} ` +
        `energy=${s.energy ?? '—'} moods=[${s.moods.join(', ')}] genre=${s.genre ?? '—'}`,
    );
  }
  console.log('');
  console.log(`AudioMuse tracks read:  ${counts.seen}`);
  console.log(`  no usable signal:     ${counts.noSignal}`);
  if (args.dryRun) {
    console.log(`  would update existing: ${counts.updated}`);
    console.log(`  would add as new:      ${counts.inserted}`);
    console.log('(dry run — nothing written)');
  } else {
    console.log(`  rows inserted:        ${counts.inserted}`);
    console.log(`  rows updated:         ${counts.updated}`);
    console.log(`  left untouched:       ${counts.skipped}`);
  }
  console.log('');
  if (args.dryRun) {
    console.log('Run again without --dry-run to import.');
    return;
  }
  console.log(
    'Done. This imported moods/bpm/key/energy/genre only — embeddings and\n' +
      'ending-aware transition data are NOT included. Next, in SUB/WAVE open\n' +
      'Admin -> Library and run an acoustic analysis pass to add them.',
  );
}

main().catch((err) => {
  console.error(`\nImport failed: ${err.message}`);
  process.exit(1);
});
