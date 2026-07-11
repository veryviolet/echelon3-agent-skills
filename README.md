# echelon3 — agent plugins

Marketplace-installable plugins that teach AI coding agents to use
[echelon3](https://github.com/veryviolet/echelon3), a config-driven PyTorch training
framework. This repo is **both** a Codex marketplace (`.agents/plugins/marketplace.json`)
and a Claude Code marketplace (`.claude-plugin/marketplace.json`); the two plugin
manifests share one skill (`plugins/echelon3/skills/echelon3/SKILL.md`).

## Codex

```bash
codex plugin marketplace add veryviolet/echelon3-agent-skills
codex plugin add echelon3@veryviolet
```

Verified: `codex plugin list` → `echelon3@veryviolet  installed, enabled`.

## Claude Code

```
/plugin marketplace add veryviolet/echelon3-agent-skills
/plugin install echelon3@veryviolet
```

Verified: `claude plugin details echelon3` → `Skills (1) echelon3`.

## Layout

```
.agents/plugins/marketplace.json         # Codex marketplace
.claude-plugin/marketplace.json          # Claude marketplace
plugins/echelon3/
  .codex-plugin/plugin.json              # Codex plugin manifest (skills: ./skills/)
  .claude-plugin/plugin.json             # Claude plugin manifest
  skills/echelon3/SKILL.md               # shared skill (the echelon3 usage guide)
echelon3-core.md                         # source of the skill content
```

Edit `echelon3-core.md`, then regenerate `SKILL.md`.
