#!/bin/bash
#============================
# Test SSH Agent key forwarding
# so it is possible to use a
# local deploy key with a repository
# on a remote node.
#
# The approach here is intentionally
# avoiding any additions to ~/.ssh/config
# since we are engaging with a repository
# on an ephemeral resource and I don't
# want to disrupt any other existing
# ~/.ssh/config settings that may be used to
# connect this resource with other workflow
# tools.
#
# The private key used here is not included
# in this repository - you will need to
# use a different deploy key with a
# different repository to successfully
# run this script.
#============================

#===========================
# Set up
#===========================

# Username; this script assumes the same username
# locally (here) and on the remote node.
user=$(whoami)

# Private key created with ssh-keygen -t ed25519
private_key="/home/${user}/.ssh/id_ed25519_dynamic-learning-rivers"

# The repository we want to pull, modify, and push back
repository="parallelworks/dynamic-learning-rivers"

# The branch of the repository we want to use
branch="test_deploy_key"

# The full path of the location to which the repo will be
# on the remote node.
abs_path_to_repo="/home/${user}/$(basename $repository)"

# Name of remote node
remote_node="gcev2.clusters.pw"

echo Checking inputs to test:
echo user: $user
echo remote_node: $remote_node
echo private_key: $private_key
echo Checking for private_key: $(ls $private_key)
echo repository: $repository
echo branch: $branch
echo abs_path_to_repo: $abs_path_to_repo

#==========================
# Run the Test
#==========================

# Remote pull
# Only the pull, clone, and push steps require the deploy key. If the
# repository already exists on the remote resource, this operation
# will simply print out a failure notice. Since there are git pull
# operations later, a failed (second) clone attempt when the clone already
# exisits is not a problem. Note that this first clone step MUST
# be done via SSH and not via HTTPS because .git/config sets the
# origin to either SSH or HTTPS during this step. If we used an
# HTTPS call here, we would not be able to make the final
# SSH-authenticated git push at the very end without changing the
# .git/config.
#
# The `ssh-agent bash -c 'ssh-add <keyfile>; ssh -A <cluster> <remote commands>` construct
# allows for a one-time addition of the key file (for each invocation)
# to the SSH Agent and a one-time forwarding of that key to the remote
# cluster to execute a remote command. Thus, the SSH Agent forwarding is
# only valid for the relatively short duration of the actual execution
# of the command instead of the whole session.
echo "=====> Clone repo to node..."
ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${user}@${remote_node} git clone git@github.com:${repository}"

# Test presence of repo
echo "=====> Test presence of repo..."
ssh $user@$remote_node ls

# Set which branch we are on. The --set-upstream and pull operations
# are needed if this branch already exists. All pulls need to be
# authenticated. The escape quoting is necessary because we need quotes
# around the two commands that need to be executed remotely: the cd and
# the git. Compare to the single command executed in the clone operation,
# above. Without the extra escaped quotes, the presence of the ";" would
# make the ssh-agent's bash interpreter think that the git pull command
# should be executed locally (here) and not on the remote node. Similar
# escape quoting is used in the commit message and the final git push.
#
# There is an alternative to including a cd command for each git
# operation; --git-dir can be used to issue git commands when outside
# of the repository. For example:
#
# ssh $user@$remote_node git --git-dir=${abs_path_to_repo}/.git branch ${branch}
#
# This option requires explicitly listing the hidden .git directory
# within the repository. This applies to older versions of git (1.8),
# the newer versions of git allow for the -C option which may be more
# straightforward to use. In particular, the --git-dir option applies
# other changes that are not clear to me.  I have not tested -C since
# this applies to newer versions of git not supported by default version
# of git available on RHEL7.
echo "======> Create and checkout a branch..."
ssh $user@$remote_node "cd ${abs_path_to_repo}; git branch ${branch}"
ssh $user@$remote_node "cd ${abs_path_to_repo}; git checkout ${branch}"
ssh $user@$remote_node "cd ${abs_path_to_repo}; git branch --set-upstream-to=origin/${branch} ${branch}"
ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${user}@${remote_node} \"cd ${abs_path_to_repo}; git pull\""

# Make a change to the repo
# (Note that this particular repo's .gitignore will ignore filenames
# that match certain patterns, in particular ".log")
ssh $user@$remote_node "echo Testing on $(date) >> ${abs_path_to_repo}/ml_models/test.std.out"

# Add and commit
# Note that git add --all will also register any deletions!!!
# Note that git add <pathspec> is interpreted as and absolute path, so you
# cannot use the "git add ." shortcut to imply the repo contents without changing
# directories first.
# Note the escaped quoting around the commit message to group spaces in the commit message.
echo "=====> Add and commit..."
ssh $user@$remote_node "cd ${abs_path_to_repo}; git add --all ."
ssh $user@$remote_node "cd ${abs_path_to_repo}; git commit -m \"Testing deploy key on $(date)\""

# Remote push
# Only the clone, pull, and push steps require the deploy key.
echo "=====> Push..."
ssh-agent bash -c "ssh-add ${private_key}; ssh -A ${user}@${remote_node} \"cd ${abs_path_to_repo}; git push origin ${branch}\""

#The `ssh-agent bash -c 'ssh-add <keyfile>; ssh -A <cluster> <remote commands>` construct 
#allows for a one-time addition of the key file to the SSH Agent and a one-time forwarding 
#of that key to the remote cluster to execute a remote command. 
