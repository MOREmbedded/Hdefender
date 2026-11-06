# Rules for this project

## Storage: F: only

- Store files only on F:. Never write temp, cache, build output, logs, scratch files or memory notes on C:.
- Never install anything on C:.
- Before running Cargo or other tools, point caches and temp at F: (`. .\dev-env.ps1` in PowerShell; in bash: `export CARGO_HOME="F:/workspace_defender/.cache/cargo" TMP="F:/workspace_defender/.cache/tmp" TEMP="F:/workspace_defender/.cache/tmp"`).
- Use `F:\workspace_defender\.cache\tmp` for temporary files, not the session scratchpad (which is on C:).
- Don't use background shell runs: the app writes their output logs to C:. Run long jobs in foreground steps, with output redirected to `F:\workspace_defender\.cache\tmp`.
- After running shell commands, move any session log files from the app's C: temp `tasks` folder to `F:\workspace_defender\.cache\claude-task-logs`.
- Prefer the file tools (Read, Glob, Grep, Edit, Write) for quick checks; they leave no logs on C:.
