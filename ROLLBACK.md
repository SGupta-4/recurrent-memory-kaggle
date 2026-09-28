# Rollback Procedures

Before a risky change:
1. Record the last known-good commit SHA/tag.
2. Record the experiment config/artifact IDs.

If reverting:
1. Revert the offending commit (or restore specific files).
2. DO NOT delete generated evidence.
3. Rerun `smoke` checks.
4. Verify the output schema and baseline metrics match.

## Known Good States
*List known good commits and configs here as they are established.*
- (Initial scaffolding) - No commit yet.
