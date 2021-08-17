#!/usr/bin/env bash

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites "condorpool" \
    --output-sites "local" \
    --cleanup "leaf" \
    --cluster "label" \
    --force \
    --submit \
    workflow.yml

