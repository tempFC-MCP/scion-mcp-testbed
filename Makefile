PYTHON ?= python3

.PHONY: simulate check-scheduler figure2 public-data ripe hosting transit-comparison scion table1 figure3 figure6 paired-outages pilot-validation

simulate: check-scheduler
	$(PYTHON) scripts/run_experiments.py
	$(PYTHON) scripts/local_hub_comparison.py
	$(PYTHON) scripts/report_experiments.py
	$(PYTHON) scripts/focused_studies.py
	$(PYTHON) scripts/check_simulation_results.py

check-scheduler:
	$(PYTHON) scripts/check_scheduler.py

figure2:
	$(PYTHON) scripts/local_hub_comparison.py --figure-only

scion:
	PYTHON="$(PYTHON)" bash experiments/scion-localnet/paper-run.sh all

table1:
	PYTHON="$(PYTHON)" bash experiments/scion-localnet/paper-run.sh delivery

figure6: table1

figure3 paired-outages pilot-validation:
	PYTHON="$(PYTHON)" bash experiments/scion-localnet/paper-run.sh $@

public-data: ripe hosting

ripe:
	$(PYTHON) scripts/fetch_topology.py
	$(PYTHON) scripts/run_experiments.py --ripe-sample data/ripe/sample.json
	$(PYTHON) scripts/report_experiments.py --ripe

hosting:
	$(PYTHON) scripts/asn_capacity.py $(if $(ASN_HTML),--source-html "$(ASN_HTML)")

transit-comparison:
	@test -n "$(BATCH)" || { echo 'Set BATCH to the batch directory from your expanded SCION run.'; exit 2; }
	$(PYTHON) scripts/transit_comparison.py --batch "$(BATCH)"
