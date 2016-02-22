#/bin/bash

set +x

git add --all
git commit -m "$1"
git push
