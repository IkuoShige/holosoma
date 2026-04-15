# Holosoma

Python ML project (PyTorch, robotics simulation). Source in `src/`, tests in `tests/`.
Lint: `ruff check .`, Types: `mypy .`, Tests: `pytest`.

## ECC + Codex Cooperation Rules

This project uses **everything-claude-code (ECC)** agents and **codex-cli** (direct).
Claude (ECC agents) and Codex (GPT) provide cross-model validation.

Codex is invoked directly via `codex exec` — **not** through the codex-plugin, which
stalls in this environment due to sandbox/bwrap issues.

### Codex CLI Conventions

```bash
# General task (read-only or write)
codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh "PROMPT"

# Review via piped diff (codex review cannot bypass sandbox)
git diff main...HEAD -- PATH | \
  codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh \
  "You are a code reviewer. Review the git diff from stdin. Output a markdown table: File, Line, Severity (CRITICAL/HIGH/MEDIUM/LOW), Finding."

# Rescue / second opinion
codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh "CONTEXT + QUESTION"
```

- Always use `--dangerously-bypass-approvals-and-sandbox` (bwrap unavailable).
- Always use `-c model=gpt-5.4 -c model_reasoning_effort=xhigh`.
- Set Bash timeout by task complexity: 120s for simple queries, 300s for reviews, 600s (max) for large diffs or rescue tasks.
- Pipe context via stdin when the prompt alone is insufficient.

### Rule 1: Dual Review

**Trigger**: User asks for review, says "check this", or significant implementation is complete.

1. Spawn `everything-claude-code:python-reviewer` agent on changed Python files.
   For non-Python files, use `everything-claude-code:code-reviewer` instead.
2. Fix any CRITICAL/HIGH issues found before proceeding.
3. Run codex review via Bash:
   ```bash
   git diff main...HEAD -- <paths> | \
     codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh \
     "You are a code reviewer. Review the git diff from stdin for correctness, bugs, and security. Output a markdown table: File, Line, Severity, Finding."
   ```
4. Present a merged findings table:
   - "Both flagged" = high confidence
   - "Claude only" / "Codex only" = cross-model catches
5. Do NOT auto-fix Codex findings. Ask the user which to address.

**Skip Codex for**: docs-only, single-line fixes, config tweaks.

### Rule 2: Dual Planning

**Trigger**: User asks for a plan, design, or architecture for a non-trivial task.

1. Spawn `everything-claude-code:planner` agent to create an initial plan
   (for architecture decisions, use `everything-claude-code:architect` instead).
2. Pipe Claude's plan to codex for adversarial review:
   ```bash
   echo "PLAN:\n<claude_plan>\n\nReview this plan. Identify blind spots, missing edge cases, better alternatives, and risks. Propose improvements." | \
     codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh -
   ```
3. Synthesize both perspectives into a final plan:
   - Where both agree = high-confidence steps
   - Where they differ = present both options with trade-offs to the user
   - Codex-only suggestions = evaluate and include if valuable
4. Present the synthesized plan to the user for approval before implementing.

**Skip Codex for**: simple tasks with obvious single approach (rename, typo fix, etc.).

### Rule 3: Rescue Escalation

**Trigger**: 3+ failed attempts at the same fix, or a domain where GPT may have better coverage.

1. Tell the user: "Escalating to Codex for a second opinion."
2. Run codex with structured context:
   ```bash
   codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh \
     "Context: <what was tried and why it failed>. Error: <exact error>. Files: <paths>. Suggest a fix."
   ```
3. Evaluate Codex's suggestion against project conventions before applying.
   Do NOT blindly apply Codex output.

### Rule 4: Cross-Verification

**Trigger**: Multi-file implementation, new module, or architectural change completed.

1. In parallel:
   - Spawn `everything-claude-code:security-reviewer` agent on changed files.
   - Run codex review via piped diff (same as Rule 1 step 3).
2. Merge and deduplicate findings.
3. Security CRITICAL = must fix. Other issues = user decides.

### Rule 5: PyTorch Error Chain

**Trigger**: Tensor shape mismatches, CUDA errors, training failures.

1. Spawn `everything-claude-code:pytorch-build-resolver` agent.
2. If it fails after 2 attempts, escalate to codex:
   ```bash
   codex exec --dangerously-bypass-approvals-and-sandbox -c model=gpt-5.4 -c model_reasoning_effort=xhigh \
     "PyTorch error. Traceback: <full traceback>. Previous attempts: <what was tried>. Suggest a fix."
   ```

### Constraints

- Do NOT invoke Codex for every small edit. Only for reviews, stuck situations, and verification.
- Do NOT run dual reviews on the same change more than once unless user asks.
- Do NOT auto-fix based on Codex output without user approval.
- If codex-cli fails, continue with Claude only and note the failure.
