#!/usr/bin/env bash

set -euo pipefail

owner="François-Guillaume Fernandez"
current_year="$(date +%Y)"
notice_pattern="^# Copyright \\(C\\) ([0-9]{4})(-([0-9]{4}))?, ${owner}\\.$"

cd "$(git rev-parse --show-toplevel)"

if ! notices="$(git grep -n -I '^# Copyright (C) ' -- '*.py')"; then
    echo "No Python copyright notices found." >&2
    exit 1
fi

while IFS=: read -r file line notice; do
    if [[ ! "${notice}" =~ ${notice_pattern} ]]; then
        echo "${file}:${line}: unexpected copyright notice: ${notice}" >&2
        exit 1
    fi

    start_year="${BASH_REMATCH[1]}"
    range="${BASH_REMATCH[2]}"
    end_year="${BASH_REMATCH[3]:-${start_year}}"

    if ((start_year < 2020 || start_year > current_year || end_year > current_year)) ||
        { [[ -n "${range}" ]] && ((end_year <= start_year)); }; then
        echo "${file}:${line}: invalid copyright years: ${notice}" >&2
        exit 1
    fi
done <<< "${notices}"

git grep -l -I '^# Copyright (C) ' -- '*.py' |
    while IFS= read -r file; do
        sed -E -i.bak \
            -e "s/^(# Copyright \\(C\\) [0-9]{4})-[0-9]{4}(, ${owner}\\.)$/\\1-${current_year}\\2/" \
            -e "/^# Copyright \\(C\\) ${current_year}, ${owner}\\.$/! s/^(# Copyright \\(C\\) [0-9]{4})(, ${owner}\\.)$/\\1-${current_year}\\2/" \
            "${file}"
        rm -f "${file}.bak"
    done

echo "Copyright notices updated to ${current_year}."
