.PHONY: setup test check package

setup:
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/ -q

check:
	.venv/bin/python scripts/validate_submission.py

package:
	zip -r submission.zip notebooks src tests queries submit requirements.txt README.md TAKEHOME.md Makefile scripts -x '*.pyc' -x '__pycache__/*'
