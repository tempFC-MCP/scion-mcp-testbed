#!/usr/bin/env bash
# Run a testbed from start to finish, or run an experiment on an existing lab.
set -eu
cd -- "$(dirname -- "$0")"

python=${PYTHON:-python3}
case "${1:-}" in
all|delivery|figure3|figure6|paired-outages|pilot-validation)
    "$python" lab.py bootstrap
    "$python" lab.py doctor
    "$python" lab.py build
    "$python" replay.py build

    run_profile() (
        profile=$1
        shift
        existing=$(docker ps -a --filter "name=clab-mcp-scion-${profile}-" --format '{{.Names}}')
        if [[ -n $existing ]]; then
            echo "The $profile lab already exists. Run individual experiments on it, or take it down first." >&2
            exit 1
        fi
        if [[ ! -f .state/${profile}-gen/mcp-scion-${profile}.clab.yml ]]; then
            "$python" lab.py generate --profile "$profile"
        fi
        trap '"$python" lab.py down --profile "$profile" --containerized-clab || exit 1' EXIT
        "$python" lab.py up --profile "$profile" --containerized-clab
        for experiment in "$@"; do
            bash paper-run.sh "$experiment"
        done
    )
    case "$1" in
        all)
            run_profile research reach validation
            run_profile expanded expanded paired
            ;;
        delivery|figure6) run_profile expanded expanded ;;
        figure3) run_profile research reach ;;
        pilot-validation) run_profile research validation ;;
        paired-outages) run_profile expanded paired ;;
    esac
    exit 0
    ;;
    reach|validation|expanded|paired) study=$1 ;;
    *) echo 'Usage: bash paper-run.sh {all|delivery|figure3|figure6|paired-outages|pilot-validation|reach|validation|expanded|paired}' >&2; exit 2 ;;
esac

mkdir -p .state/reproductions
run=$(mktemp -d ".state/reproductions/${study}.XXXXXXXX")
snapshot="$run/observations"
batch="$run/batch"
echo "Saving inputs, measurements and plots in $run"
case "$study" in
    reach)
        "$python" network.py shape --profile continental --rate-mbit 1000 --jitter-ms 0.05
        "$python" network.py collect --out "$snapshot" --count 100
        "$python" coordination.py prepare --snapshot "$snapshot" --out "$batch"
        "$python" coordination_report.py --out "$batch"
        ;;
    validation)
        "$python" replay.py install
        "$python" network.py shape --profile wan --rate-mbit 100
        "$python" network.py collect --out "$snapshot" --count 100
        "$python" batch.py prepare --snapshot "$snapshot" --out "$batch"
        "$python" batch.py run --out "$batch"
        "$python" report.py "$batch"
        ;;
    paired)
        "$python" expanded_study.py install
        "$python" network.py shape --lab-profile expanded --profile continental --rate-mbit 1000 --jitter-ms 0.05
        "$python" network.py collect --lab-profile expanded --out "$snapshot" --count 50
        "$python" paired_outage.py prepare --snapshot "$snapshot" --out "$batch"
        "$python" paired_outage.py run --out "$batch"
        "$python" paired_outage_report.py --out "$batch"
        ;;
    expanded)
        "$python" expanded_study.py install
        "$python" network.py shape --lab-profile expanded --profile continental --rate-mbit 1000 --jitter-ms 0.05
        "$python" network.py collect --lab-profile expanded --out "$snapshot" --count 50
        "$python" expanded_study.py prepare --snapshot "$snapshot" --out "$batch"
        "$python" check_expanded.py --out "$batch"
        "$python" coverage_followup.py peers --out "$batch"
        "$python" resume_timing.py --out "$batch"
        "$python" strict_local.py --source "$batch" --out "$run/strict"
        "$python" network.py collect --lab-profile expanded --out "$run/fresh-observations" --count 50
        "$python" coverage_followup.py prepare-peer --source "$batch" --snapshot "$run/fresh-observations" --out "$run/peer180"
        "$python" coverage_followup.py peers --out "$run/peer180"
        "$python" coverage_followup.py prepare-allocations --source "$batch" --out "$run/allocations"
        "$python" expanded_report.py --out "$batch" --strict-out "$run/strict" --coverage-out "$run/allocations" --peer-out "$run/peer180"
        "$python" ../../scripts/transit_comparison.py --batch "$batch"
        ;;
esac
echo "Report: $PWD/$batch/report"
