#!/bin/sh
set -e

harness_root="$(python3 -c 'import os; import determined as x; print(os.path.dirname(x.__file__))')"
common_root="$(python3 -c 'import os; import determined_common as x; print(os.path.dirname(x.__file__))')"
cli_root="$(python3 -c 'import os; import determined_cli as x; print(os.path.dirname(x.__file__))')"

set -x

cp "hotpatch-0.14.2/harness/determined/_train_context.py" "$harness_root/_train_context.py"
cp "hotpatch-0.14.2/harness/determined/pytorch/__init__.py" "$harness_root/pytorch/__init__.py"
cp "hotpatch-0.14.2/harness/determined/pytorch/_data.py" "$harness_root/pytorch/_data.py"
cp "hotpatch-0.14.2/harness/determined/pytorch/_pytorch_context.py" "$harness_root/pytorch/_pytorch_context.py"
cp "hotpatch-0.14.2/harness/determined/pytorch/_pytorch_trial.py" "$harness_root/pytorch/_pytorch_trial.py"
cp "hotpatch-0.14.2/harness/determined/pytorch/samplers.py" "$harness_root/pytorch/samplers.py"
