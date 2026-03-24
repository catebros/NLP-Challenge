VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: all setup run clean

all: setup run

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --quiet torch numpy
	@echo "Virtual environment ready."

run: $(VENV)/bin/activate
	$(PYTHON) main.py

clean:
	rm -rf $(VENV) __pycache__
