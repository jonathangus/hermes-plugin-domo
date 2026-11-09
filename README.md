# Domo plugin for Hermes

The backend of [Domo](https://github.com/jonathangus), a chief-of-staff app for [Hermes Agent](https://hermes-agent.nousresearch.com): a Hermes plugin that adds agent tools and serves the routes a Domo app (web, iPhone, Mac) talks to.

It gives each agent:

- Tools: `domo_ask` (decision cards before anything outbound, paying or deleting), `domo_todo`, `domo_receipt`, `domo_run`, `domo_brief`, `domo_fact`, `domo_suggest`, `domo_memory`, `domo_open`, and `domo_shared` (shared notes between agents).
- Dashboard routes under `/api/plugins/domo/*`: decisions, to-dos, runs, receipts, spaces, chat and session passthrough to the profile's API server, routines (cron), files, memory, search.
- A per-profile SQLite store in `<profile>/plugin-data/domo/`.

```
Domo app ─▶ Hermes dashboard :9119 (login) ─ /api/plugins/domo/* ─▶ this plugin
                                                   └─▶ Hermes API server :8642 (loopback, Bearer key, /p/<profile>/)
```

## Install

The plugin must live in a directory named `domo` (its routes import `domo` as a package).

```bash
git clone https://github.com/jonathangus/hermes-plugin-domo ~/.hermes/plugins/domo
# a named profile: ~/.hermes/profiles/<name>/plugins/domo
```

Then, per profile, in `config.yaml`:

```yaml
plugins:
  enabled: [domo]
tools:
  tool_search:
    enabled: "off"        # otherwise the domo tools are deferred and the agent doesn't see them
```

and enable the toolset for the platforms you use:

```bash
hermes [-p <profile>] tools enable domo --platform api_server   # also: cli, telegram, whatsapp, slack
```

## Run it so an app can connect

In the default profile's `~/.hermes/.env`:

```
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=domo
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=<scrypt hash from Hermes's plugins.dashboard_auth.basic.hash_password>
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<random string>
```

In every profile's `.env` (default included): `API_SERVER_KEY=<random string>`. With more than one profile, set `gateway.multiplex_profiles: true` in the default profile's `config.yaml`, so one API port serves each profile under `/p/<name>/`.

Start the gateway (`hermes gateway run`) and the dashboard (`hermes dashboard --host 0.0.0.0 --port 9119 --no-open`), reach it over Tailscale at `http://<host>.<tailnet>.ts.net:9119`, and sign in from the app.

Optional per profile: `plugin-data/domo/profile.json` (`display_name`, `kind`, `spaces`) for display names and starting spaces; `DOMO_SPACES` (comma-separated) in the profile's `.env`.

## Optional extras

- Search by meaning: any OpenAI-compatible embeddings endpoint (`DOMO_EMBED_URL`, `DOMO_EMBED_MODEL`, `DOMO_EMBED_KEY`), e.g. Ollama with `embeddinggemma-2:270m`. Without it search matches words only.
- Auto-filing chats into spaces: `TYPESAFE_API_KEY`. Without it chats stay where you put them.
- `rtk` on PATH: compresses terminal output before it reaches the model.
- `DOMO_REPO_ROOTS`: colon-separated folders the app scans for git repos.

## Compatibility

Built against Hermes `_config_version` 52. The plugin reads parts of Hermes that aren't a stable API (`hermes_constants`, `state.db` tables, API server session/chat-stream endpoints), so other versions may need changes.

## Licence

MIT
