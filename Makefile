PY ?= .venv/bin/python
LAKE ?= lake
BOOTSTRAP ?= localhost:9092

.PHONY: install lint typecheck test test-kafka check up down demo kafka-demo dashboard bench clean

install:  ## create .venv with all extras
	python3.12 -m venv .venv && $(PY) -m pip install -q -U pip && $(PY) -m pip install -q -e ".[dev,dashboard]"

lint:
	$(PY) -m ruff check src tests infra benchmarks scripts dashboard && $(PY) -m ruff format --check src tests infra benchmarks scripts dashboard

typecheck:
	$(PY) -m mypy src

test:  ## unit + in-process end-to-end tests (no Kafka needed)
	$(PY) -m pytest

test-kafka:  ## integration tests against the broker from `make up`
	FLEETSCOPE_BOOTSTRAP=$(BOOTSTRAP) $(PY) -m pytest -m kafka

check: lint typecheck test

up:  ## Kafka + Prometheus + Grafana
	docker compose up -d --wait kafka prometheus grafana && docker compose up topics

down:
	docker compose down -v

demo:  ## everything in-process: replay -> lake -> scenarios -> selection -> evaluation
	rm -rf $(LAKE) && $(PY) -m fleetscope.cli demo --lake $(LAKE) --vehicles 200 --minutes 20 \
		--comma2k19 data/comma2k19_sample --clones 50 --no-fsync --budget-gb 5

kafka-demo: up  ## the same through Kafka (fresh topic per run), Prometheus metrics on :9108
	rm -rf $(LAKE)
	$(eval TOPIC := demo.$(shell date +%s))
	docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 \
		--create --topic $(TOPIC) --partitions 12 --config message.timestamp.type=LogAppendTime
	$(PY) -m fleetscope.cli replay --lake $(LAKE) --bootstrap $(BOOTSTRAP) --topic $(TOPIC) \
		--vehicles 200 --minutes 20 --comma2k19 data/comma2k19_sample --clones 50
	$(PY) -m fleetscope.cli process --lake $(LAKE) --bootstrap $(BOOTSTRAP) --topic $(TOPIC) \
		--metrics-port 9108 --exit-when-idle 5 --drain-on-exit --group $(TOPIC)
	$(PY) -m fleetscope.cli mine --lake $(LAKE) && $(PY) -m fleetscope.cli select --lake $(LAKE) --budget-gb 5

dashboard:
	FLEETSCOPE_LAKE=$(LAKE) $(PY) -m streamlit run dashboard/app.py

bench:  ## add BENCH_ARGS="--bootstrap localhost:9092" to include the Kafka path
	$(PY) benchmarks/run.py --out docs/benchmarks.md $(BENCH_ARGS)

clean:
	rm -rf $(LAKE) .pytest_cache .mypy_cache .ruff_cache
