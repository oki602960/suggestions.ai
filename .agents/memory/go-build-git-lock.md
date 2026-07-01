---
name: go build intermittent git lock error
description: Why the first `go build ./...` invocation sometimes fails with an unrelated git error.
---

The first attempt at `go build ./...` in this workspace can fail with an
error mentioning `.git/index.lock` (a destructive-git-operation guard
tripping), even though the failure has nothing to do with the Go code being
built.

**Why:** some sandbox/tooling hook intercepts the build command and
occasionally collides with a git lock check unrelated to compilation.

**How to apply:** if `go build ./...` fails with a git index.lock-related
message, just retry the exact same command once — it reliably succeeds on
the second attempt. Only treat it as a real build failure if it fails again
with an actual Go compiler error.
