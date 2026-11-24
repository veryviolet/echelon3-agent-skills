# echelon3 — agent plugins

Marketplace-installable plugins that teach AI coding agents to use
[echelon3](https://github.com/veryviolet/echelon3), a config-driven PyTorch training
framework. One skill (`plugins/echelon3/skills/echelon3/SKILL.md`) is published
through four marketplaces: Codex (`.agents/plugins/marketplace.json`), Claude Code
(`.claude-plugin/marketplace.json`), Grok (`.grok-plugin/marketplace.json`), and
Kimi Code (`kimi.plugin.json`).

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

## Grok

```bash
grok plugin marketplace add veryviolet/echelon3-agent-skills
grok plugin install echelon3 --trust
```

Verified: `grok plugin details echelon3` → `echelon3 v0.1.23`, 1 skill dir.

## Kimi Code

```bash
/plugins install https://github.com/veryviolet/echelon3-agent-skills
/reload
```

The root `kimi.plugin.json` exposes the same shared skill to Kimi Code.

## Layout

```
kimi.plugin.json                         # Kimi Code plugin manifest
.agents/plugins/marketplace.json         # Codex marketplace
.claude-plugin/marketplace.json          # Claude marketplace
.grok-plugin/marketplace.json            # Grok marketplace
plugins/echelon3/
  .codex-plugin/plugin.json              # Codex plugin manifest (skills: ./skills/)
  .claude-plugin/plugin.json             # Claude plugin manifest
  .grok-plugin/plugin.json               # Grok plugin manifest
  skills/echelon3/SKILL.md               # shared skill (the echelon3 usage guide)
echelon3-core.md                         # source of the skill content
```

Plugin manifests share one version. Edit `echelon3-core.md`, then copy that body
under the existing frontmatter of `SKILL.md`.
