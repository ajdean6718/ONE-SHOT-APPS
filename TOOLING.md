# Claude Code tooling for this repository

This file records the skills, plugins, and MCP servers configured for Claude Code
cloud sessions in this repository. Everything below is loaded automatically from
`.claude/settings.json`, `.claude/skills/`, `.claude/commands/`, and `.mcp.json`.

## Plugins (installed from marketplaces, declared in `.claude/settings.json`)

| Plugin | Marketplace | Source repo | Provides |
| --- | --- | --- | --- |
| frontend-design | claude-plugins-official | anthropics/claude-code `plugins/frontend-design` | `frontend-design` skill |
| feature-dev | claude-plugins-official | anthropics/claude-code `plugins/feature-dev` | `/feature-dev` workflow, `code-explorer`, `code-architect`, `code-reviewer` agents |
| superpowers | claude-plugins-official | obra/superpowers | 15 workflow skills (brainstorming, writing-plans, TDD, systematic-debugging, ...) |
| ui-ux-pro-max | ui-ux-pro-max-skill | nextlevelbuilder/ui-ux-pro-max-skill | 7 design skills plus the searchable design database |
| ecc | ecc | affaan-m/everything-claude-code | 397 skills, 71 agents, 8 hooks, `chrome-devtools` MCP server |

ECC is configured with `hooks_enabled=true` and `hook_profile=standard`. It adds
roughly 46k always-on tokens to every session. Disable it in
`.claude/settings.json` if that cost is not worth it for a given build.

## Project skills (`.claude/skills/`)

| Skill | Source repo | Notes |
| --- | --- | --- |
| reverse-engineer-anything | morluto/rea | Pairs with the `rea` MCP server below. Static JavaScript/Electron/web analysis works here. Native binary analysis needs Hopper, Ghidra, or IDA, none of which are installed. |
| clone-website | JCodesMore/ai-website-cloner-template | Invoked as `/clone-website <url>`. Written for a Next.js 16 + shadcn/ui + Tailwind v4 app, so create that scaffold first. Needs a browser (Playwright MCP below). |
| vercel-react-best-practices | vercel-labs/agent-skills | Installed with `npx skills add`, tracked in `skills-lock.json`. |
| web-design-guidelines | vercel-labs/agent-skills | Installed with `npx skills add`, tracked in `skills-lock.json`. |
| changelog-generator, competitive-ads-extractor, content-research-writer, developer-growth-analysis, domain-name-brainstormer, file-organizer, image-enhancer, invoice-organizer, langsmith-fetch, lead-research-assistant, meeting-insights-analyzer, raffle-winner-picker, tailored-resume-generator, twitter-algorithm-optimizer, video-downloader, webapp-testing | ComposioHQ/awesome-claude-skills | Every skill from that repo that is not already provided by the account's claude.ai skills and that does not require a Composio API key. |

Skipped from ComposioHQ/awesome-claude-skills: `connect`, `connect-apps`,
`connect-apps-plugin`, `skill-share`, and the 832 `composio-skills/*` entries
(all need a Composio or Rube API key), `template-skill` (placeholder), and
`artifacts-builder`, `brand-guidelines`, `canvas-design`, `internal-comms`,
`mcp-builder`, `skill-creator`, `slack-gif-creator`, `theme-factory`,
`document-skills` (already enabled as claude.ai skills on this account).

## MCP servers (`.mcp.json`)

| Server | Package | Notes |
| --- | --- | --- |
| playwright | `@playwright/mcp@latest` (microsoft/playwright-mcp) | Runs headless with the sandbox off and the container's preinstalled Chromium at `/opt/pw-browsers/chromium`. The stock config fails here because no Chrome channel is installed. |
| rea | `rea-agents@latest` (morluto/rea) | 139 tools. Also available as the `rea` CLI (`npm install -g rea-agents`). |

## Update commands

```bash
claude plugin marketplace update            # refresh all marketplaces
claude plugin update <plugin>@<marketplace> # update one plugin
npx skills update                           # update the Vercel skills
```
