# team_results

Small result files (reports, plots, CSVs) made on one machine and shared with
the team through git. `results/` itself is gitignored and per-machine.

- Add files only with `scripts/package_results.py <label> --machine <name> <files...>`.
  It creates `team_results/<date>_<label>/` with a `provenance.json` (git commit,
  branch, dirty flag, machine), never overwrites an earlier package, and refuses
  videos and files over 5 MB.
- **No videos and no images of people, ever.** Recordings go to the private shared
  drive (`docs/RECORDING_GUIDE.md`, section 6).
- Treat a package as a record: don't edit it after committing. Package a new run instead.
