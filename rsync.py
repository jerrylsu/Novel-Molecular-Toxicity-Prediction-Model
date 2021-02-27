#!/usr/bin/env python3

import subprocess as sp
from typing import Optional, List


class Rsync(object):
    def __init__(self, source_dir:str, target_dir: str):
        self.server_user_name = "lsu1"
        self.server_ip = "139.224.58.222"
        self.source_dir = source_dir
        self.target_dir = target_dir

    def rsync(self,
              exclude: Optional[List[str]],
              dry_run: Optional[bool]=False):
        # Rsync files
        exclude = ' '.join([f'--exclude={ele}' for ele in exclude])
        cmd = f'rsync -avhi{"n" if dry_run else ""} --progress --delete {exclude} \
                {self.source_dir} {self.server_user_name}@{self.server_ip}:{self.target_dir}'
        print(cmd)
        sp.run(cmd, shell=True, check=True)


class Git(object):
    def __init__(self):
        self.server_user_name = "lsu1"
        self.server_ip = "139.224.58.222"

    def git_status_remote(self, target_dir: str):
        cmd = f'ssh {self.server_user_name}@{self.server_ip} "cd {target_dir}; git status"'
        sp.run(cmd, shell=True, check=True)


rsync = Rsync(source_dir="../Novel-Molecular-Toxicity-Prediction-Model/", target_dir="/home/Novel-Molecular-Toxicity-Prediction-Model/")
rsync.rsync(exclude=['.git', 'model'], dry_run=False)

git = Git()
git.git_status_remote(target_dir="/home/Novel-Molecular-Toxicity-Prediction-Model/")
# rsync -avhi --progress --delete --exclude=".git" multi-turn-dialogue-ir/ lsu1@139.224.58.222:/home/UtteranceRewriter/
