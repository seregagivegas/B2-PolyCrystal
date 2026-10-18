#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Создайте окружение по инструкции README.md: python3.12 -m venv .venv"
  read "?Нажмите Enter для закрытия"
  exit 1
fi
exec .venv/bin/python main.py
