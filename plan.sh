#!/usr/bin/env bash

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites "condorpool" \
    --output-sites "local" \
    --cleanup "leaf" \
    --force \
    --submit \
    workflow.yml

