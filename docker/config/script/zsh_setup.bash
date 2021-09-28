#!/bin/bash

mv ./zsh/powerlevel10k /home/${USER}/
mv ./zsh/.p10k.zsh /home/${USER}/
cat ./zsh/zshrc/* >> /home/${USER}/.zshrc
